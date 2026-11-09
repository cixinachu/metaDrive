"""Alternating dual-policy PPO / Social-AIRL training loop.

The update ordering, reward recovery, discriminator labels and PPO objective
follow the public main_helbing.py + PPO.py. This loop expects explicitly paired
expert transitions built from source-verified ELCV/RFV/leader role assignments.
"""
from dataclasses import dataclass
from pathlib import Path
import random
import json
import numpy as np
import torch
from torch import nn

from rl.srairl.config import SRAIRLConfig
from rl.srairl.discriminator import AnisotropicInteractionPotential, SocialAIRLDiscrim
from rl.srairl.ppo import PPO
from rl.srairl import reward_config as defaults
from env.tasks.srairl_env import SRAIRLEnv


def counterfactual_preference_mask(states, config, *, gap_reduction_m=5.0):
    """Select expert states whose gap-reduced counterpart violates safety screens."""
    frames=states.reshape(-1,config.history_len,config.frame_state_dim)
    latest=frames[:,-1]
    gap_m=latest[:,5]*config.pos_long_scale
    closing_mps=(-latest[:,6]*config.vel_scale).clamp_min(0.)
    rfv_speed_mps=(latest[:,2]*config.vel_scale+config.vel_offset-
                   latest[:,6]*config.vel_scale).clamp_min(.1)
    cf_gap=(gap_m-float(gap_reduction_m)).clamp_min(0.)
    expert_ttc=torch.where(closing_mps>1e-3,(gap_m-4.5).clamp_min(0.)/closing_mps,
                           torch.full_like(gap_m,float('inf')))
    cf_ttc=torch.where(closing_mps>1e-3,(cf_gap-4.5).clamp_min(0.)/closing_mps,
                       torch.full_like(gap_m,float('inf')))
    expert_unsafe=(gap_m<=5.)|(expert_ttc<=1.5)|(gap_m/rfv_speed_mps<=.5)
    counterfactual_unsafe=(cf_gap<=5.)|(cf_ttc<=1.5)|(cf_gap/rfv_speed_mps<=.5)
    counterfactual=frames.clone()
    counterfactual[:,:,5]=(counterfactual[:,:,5]-float(gap_reduction_m)/config.pos_long_scale).clamp_min(-1.)
    return counterfactual.reshape_as(states),counterfactual_unsafe & ~expert_unsafe


@dataclass(frozen=True)
class SRAIRLTrainingConfig:
    episodes: int = defaults.MAX_EPISODES
    action_scale: float = 1.5
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    discriminator_lr: float = 3e-4
    ppo_epochs: int = 80
    clip_epsilon: float = 0.2
    initial_action_std: float = 0.2
    action_std_decay: float = 0.05
    minimum_action_std: float = 0.1
    decay_every_steps: int = 100000
    discriminator_every_episodes: int = 150
    discriminator_batch_size: int = 64
    discriminator_updates: int = 10
    lambda_social: float = 0.0
    lambda_equilibrium: float = 0.0
    discriminator_label_smoothing: float = 0.0
    lambda_rank: float = defaults.RANK_LAMBDA
    preference_margin: float = 0.01
    seed: int = defaults.SEED
    device: str = "auto"

    def __post_init__(self):
        if not 0 <= self.discriminator_label_smoothing < 0.5:
            raise ValueError('discriminator_label_smoothing must be in [0, 0.5)')
        if self.episodes < 1 or self.action_scale <= 0 or self.ppo_epochs < 1:
            raise ValueError("episodes/action_scale/ppo_epochs must be positive")
        if self.discriminator_every_episodes < 1 or self.discriminator_batch_size < 2 or self.discriminator_updates < 1:
            raise ValueError("invalid discriminator update schedule")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if self.lambda_rank < 0 or self.preference_margin < 0:
            raise ValueError("preference-ranking weight and margin must be nonnegative")


class ExpertTransitions:
    def __init__(self, transitions, device):
        pairs = {}
        for transition in transitions:
            pairs.setdefault((transition.scenario_id, transition.step_index), {})[transition.agent] = transition
        pairs = [pair for pair in pairs.values() if "elcv" in pair and "rfv" in pair]
        if not pairs:
            raise ValueError("no paired ELCV/RFV expert transitions were supplied")
        for pair in pairs:
            a, b = pair["elcv"], pair["rfv"]
            if not (np.array_equal(a.state, b.state) and np.array_equal(a.next_state, b.next_state) and a.done == b.done):
                raise ValueError("paired expert transitions must share state, next_state and done")
        self.states = torch.as_tensor(np.stack([p["elcv"].state for p in pairs]), dtype=torch.float32, device=device)
        self.next_states = torch.as_tensor(np.stack([p["elcv"].next_state for p in pairs]), dtype=torch.float32, device=device)
        self.dones = torch.as_tensor([p["elcv"].done for p in pairs], dtype=torch.float32, device=device)
        self.actions_elcv = torch.as_tensor(np.stack([p["elcv"].action for p in pairs]), dtype=torch.float32, device=device)
        self.actions_rfv = torch.as_tensor(np.stack([p["rfv"].action for p in pairs]), dtype=torch.float32, device=device)

    def sample(self, batch_size):
        idx = torch.randint(len(self.states), (batch_size,), device=self.states.device)
        return {"states": self.states[idx], "next_states": self.next_states[idx], "dones": self.dones[idx],
                "actions_elcv": self.actions_elcv[idx], "actions_rfv": self.actions_rfv[idx]}


class SRAIRLTrainer:
    def __init__(self, envs, expert_transitions, config=None, training_config=None, output_dir=None,
                 env_factory=None, environment_count=1, dynamics_mode=None):
        self.envs = list(envs or [])
        if env_factory is None and not self.envs:
            raise ValueError("at least one SRAIRLEnv or an environment factory is required")
        if env_factory is not None and self.envs:
            raise ValueError("provide reusable environments or an environment factory, not both")
        if any(not isinstance(env, SRAIRLEnv) for env in self.envs):
            raise TypeError("SR-AIRL trainer accepts only the two-actor SRAIRLEnv")
        if any(env.dynamics_modes and env.observer.history_len != (config or SRAIRLConfig.from_mapping()).history_len for env in self.envs):
            raise ValueError("all interaction environments must use configured history length")
        self.env_factory=env_factory
        self.environment_count=max(1,int(environment_count)) if env_factory is not None else len(self.envs)
        self.dynamics_mode=dynamics_mode or (self.envs[0].dynamics_mode if self.envs else "metadrive_physics")
        self.last_env=self.envs[0] if self.envs else None
        self.config = config or SRAIRLConfig.from_mapping()
        self.train_config = training_config or SRAIRLTrainingConfig()
        if self.train_config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.train_config.device)
        torch.manual_seed(self.train_config.seed)
        np.random.seed(self.train_config.seed)
        random.seed(self.train_config.seed)
        self.expert = ExpertTransitions(expert_transitions, self.device)
        state_dim = self.config.frame_state_dim * self.config.history_len
        self.policy_elcv = PPO(state_dim, 2, self.train_config.actor_lr, self.train_config.critic_lr,
            self.config.gamma, self.train_config.ppo_epochs, self.train_config.clip_epsilon,
            self.train_config.initial_action_std, self.device)
        self.policy_rfv = PPO(state_dim, 2, self.train_config.actor_lr, self.train_config.critic_lr,
            self.config.gamma, self.train_config.ppo_epochs, self.train_config.clip_epsilon,
            self.train_config.initial_action_std, self.device)
        self.shared_potential = AnisotropicInteractionPotential(self.config.potential_longitudinal_scale,
            self.config.potential_lateral_scale, self.config.direction_lambda,
            pos_lat_scale=self.config.pos_lat_scale,pos_long_scale=self.config.pos_long_scale,
            vel_scale=self.config.vel_scale,vel_offset=self.config.vel_offset,
            prediction_horizon=self.config.interaction_prediction_horizon_s,
            prediction_steps=self.config.interaction_prediction_steps,
            prediction_dt=self.config.dt_s).to(self.device)
        self.discriminator_elcv = SocialAIRLDiscrim(state_dim, self.config.gamma, self.shared_potential,
            self.config.alpha_init, role="lcv", history_len=self.config.history_len,
            dt=self.config.dt_s,acceleration_scale=self.config.acceleration_scale,
            jerk_scale=self.config.jerk_scale).to(self.device)
        self.discriminator_rfv = SocialAIRLDiscrim(state_dim, self.config.gamma, self.shared_potential,
            self.config.alpha_init, role="fv", history_len=self.config.history_len,
            dt=self.config.dt_s,acceleration_scale=self.config.acceleration_scale,
            jerk_scale=self.config.jerk_scale).to(self.device)
        parameters = list(self.discriminator_elcv.private_g.parameters()) + list(self.discriminator_elcv.h.parameters()) + [self.discriminator_elcv.alpha_raw]
        parameters += list(self.discriminator_rfv.private_g.parameters()) + list(self.discriminator_rfv.h.parameters()) + [self.discriminator_rfv.alpha_raw]
        parameters += list(self.shared_potential.parameters())
        self.discriminator_optimizer = torch.optim.Adam(parameters, lr=self.train_config.discriminator_lr)
        self.bce = nn.BCEWithLogitsLoss()
        self.output_dir = Path(output_dir) if output_dir else None
        self.global_steps = 0
        self.episode = 0
        self.next_decay_step = self.train_config.decay_every_steps
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_episode(self, env):
        observations, _ = env.reset()
        state = observations["elcv"]
        tracking_return = 0.0
        while True:
            state_t, action_e, log_e, value_e = self.policy_elcv.select_action(state)
            _, action_f, log_f, value_f = self.policy_rfv.select_action(state)
            next_observations, info = env.step({
                "elcv": action_e.squeeze(0).detach().cpu().numpy() * self.train_config.action_scale,
                "rfv": action_f.squeeze(0).detach().cpu().numpy() * self.train_config.action_scale,
            })
            next_state = next_observations["elcv"]
            done = bool(info["done"])
            tracking_reward = float(info["tracking_reward"])
            next_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).reshape(1, -1)
            self.policy_elcv.buffer.add(state_t, action_e, log_e, value_e, tracking_reward, done, next_t)
            self.policy_rfv.buffer.add(state_t, action_f, log_f, value_f, tracking_reward, done, next_t)
            tracking_return += tracking_reward
            self.global_steps += 1
            state = next_state
            while self.global_steps >= self.next_decay_step:
                self.policy_elcv.decay_action_std(self.train_config.action_std_decay, self.train_config.minimum_action_std)
                self.policy_rfv.decay_action_std(self.train_config.action_std_decay, self.train_config.minimum_action_std)
                self.next_decay_step += self.train_config.decay_every_steps
            if done:
                return tracking_return

    @staticmethod
    def _finite(*tensors):
        return all(torch.isfinite(t).all().item() for t in tensors)

    def _discriminator_update(self):
        d = self.train_config
        losses, accuracies, private_values, potentials, rank_losses = [], [], [], [], []
        rank_pair_counts=[]
        self.policy_elcv.policy.eval(); self.policy_rfv.policy.eval()
        for _ in range(d.discriminator_updates):
            gen_e = self.policy_elcv.buffer.sample(d.discriminator_batch_size)
            gen_f = self.policy_rfv.buffer.sample(d.discriminator_batch_size)
            exp = self.expert.sample(d.discriminator_batch_size)
            if not self._finite(gen_e["states"], gen_f["states"], exp["states"], gen_e["next_states"], exp["next_states"]):
                continue
            with torch.no_grad():
                log_exp_e, _, _ = self.policy_elcv.policy.evaluate(exp["states"], exp["actions_elcv"] / d.action_scale)
                log_exp_f, _, _ = self.policy_rfv.policy.evaluate(exp["states"], exp["actions_rfv"] / d.action_scale)
            logits_gen_e = self.discriminator_elcv.discriminator_logits(
                gen_e["states"], gen_e["dones"], gen_e["logprobs"], gen_e["next_states"])
            logits_exp_e = self.discriminator_elcv.discriminator_logits(
                exp["states"], exp["dones"], log_exp_e, exp["next_states"])
            logits_gen_f = self.discriminator_rfv.discriminator_logits(
                gen_f["states"], gen_f["dones"], gen_f["logprobs"], gen_f["next_states"])
            logits_exp_f = self.discriminator_rfv.discriminator_logits(
                exp["states"], exp["dones"], log_exp_f, exp["next_states"])
            loss = self.bce(logits_exp_e, torch.full_like(logits_exp_e,1.-d.discriminator_label_smoothing))
            loss += self.bce(logits_exp_f, torch.full_like(logits_exp_f,1.-d.discriminator_label_smoothing))
            loss += self.bce(logits_gen_e, torch.full_like(logits_gen_e,d.discriminator_label_smoothing))
            loss += self.bce(logits_gen_f, torch.full_like(logits_gen_f,d.discriminator_label_smoothing))
            social = loss.new_zeros(()); equilibrium = loss.new_zeros(()); rank = loss.new_zeros(())
            if d.lambda_social > 0:
                _, private_e, _ = self.discriminator_elcv.get_reward(exp["states"])
                _, private_f, _ = self.discriminator_rfv.get_reward(exp["states"])
                social = private_e.square().mean() + private_f.square().mean()
            if d.lambda_equilibrium > 0:
                adv_e = self.discriminator_elcv.f(exp["states"], exp["dones"], exp["next_states"]).clamp(-5,5)
                adv_f = self.discriminator_rfv.f(exp["states"], exp["dones"], exp["next_states"]).clamp(-5,5)
                equilibrium = adv_e.square().mean() + adv_f.square().mean()
            rank_pairs=0
            if d.lambda_rank > 0:
                cf_states,eligible=counterfactual_preference_mask(exp['states'],self.config)
                rank_pairs=int(eligible.sum().item())
                if rank_pairs:
                    expert_e=self.discriminator_elcv.get_reward(exp['states'])[0]
                    cf_e=self.discriminator_elcv.get_reward(cf_states)[0]
                    expert_f=self.discriminator_rfv.get_reward(exp['states'])[0]
                    cf_f=self.discriminator_rfv.get_reward(cf_states)[0]
                    rank_e=nn.functional.softplus(self.train_config.preference_margin-(expert_e-cf_e)).reshape(-1)
                    rank_f=nn.functional.softplus(self.train_config.preference_margin-(expert_f-cf_f)).reshape(-1)
                    rank=(rank_e[eligible].mean()+rank_f[eligible].mean())
                rank_losses.append(float(rank.detach().cpu()))
                rank_pair_counts.append(rank_pairs)
            total = loss + d.lambda_social*social + d.lambda_equilibrium*equilibrium + d.lambda_rank*rank
            self.discriminator_optimizer.zero_grad(); total.backward(); self.discriminator_optimizer.step()
            losses.append(float(total.detach().cpu()))
            accuracies.append((float((logits_exp_e>0).float().mean()), float((logits_gen_e<0).float().mean()),
                               float((logits_exp_f>0).float().mean()), float((logits_gen_f<0).float().mean())))
            with torch.no_grad():
                _, _, phi = self.discriminator_elcv.get_reward(exp["states"])
                private_values.append(float(self.discriminator_elcv.get_reward(exp["states"])[1].mean().cpu()))
                potentials.append(float(phi.mean().cpu()))
        self.policy_elcv.policy.train(); self.policy_rfv.policy.train()
        if not losses:
            raise RuntimeError("all discriminator minibatches contained invalid values")
        # Recover AIRL reward for every accumulated policy transition, as in upstream.
        for policy, disc in ((self.policy_elcv, self.discriminator_elcv), (self.policy_rfv, self.discriminator_rfv)):
            data = policy.buffer.tensors()
            recovered = disc.calculate_policy_reward(
                data["states"], data["dones"], data["logprobs"], data["next_states"]).squeeze(-1)
            policy.buffer.rewards = torch.clamp(recovered, -10., 10.).detach().cpu().tolist()
        return {"discriminator_loss": float(np.mean(losses)),
                "expert_accuracy_elcv": float(np.mean([a[0] for a in accuracies])),
                "policy_accuracy_elcv": float(np.mean([a[1] for a in accuracies])),
                "expert_accuracy_rfv": float(np.mean([a[2] for a in accuracies])),
                "policy_accuracy_rfv": float(np.mean([a[3] for a in accuracies])),
                "private_reward": float(np.mean(private_values)), "shared_potential": float(np.mean(potentials)),
                "preference_rank_loss": float(np.mean(rank_losses)) if rank_losses else 0.0,
                "preference_pairs_used": float(np.mean(rank_pair_counts)) if rank_pair_counts else 0.0,
                "alpha_elcv": float(self.discriminator_elcv.alpha.detach().cpu()),
                "alpha_rfv": float(self.discriminator_rfv.alpha.detach().cpu()),
                "potential_L_long_m": float(self.shared_potential.L_long.detach().cpu()),
                "potential_L_lat_m": float(self.shared_potential.L_lat.detach().cpu()),
                "potential_dir_lambda": float(self.shared_potential.dir_lambda.detach().cpu())}

    def save(self, directory=None):
        directory = Path(directory) if directory else self.output_dir
        if directory is None:
            raise ValueError("output directory is required to save checkpoints")
        directory.mkdir(parents=True, exist_ok=True)
        self.policy_elcv.save(directory / "policy_elcv.pt")
        self.policy_rfv.save(directory / "policy_rfv.pt")
        torch.save(self.discriminator_elcv.state_dict(), directory / "discriminator_elcv.pt")
        torch.save(self.discriminator_rfv.state_dict(), directory / "discriminator_rfv.pt")
        torch.save({
            "episode": self.episode,
            "global_steps": self.global_steps,
            "next_decay_step": self.next_decay_step,
            "policy_elcv_optimizer": self.policy_elcv.optimizer.state_dict(),
            "policy_rfv_optimizer": self.policy_rfv.optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "seed": self.train_config.seed,
        }, directory / "training_state.pt")
        (directory / "config.json").write_text(json.dumps({"algorithm": "SR-AIRL", "reward_version":"temporal-dynamic-v2",
            "source_dataset_revision":getattr(self,"source_dataset_revision",None),
            "reference_frame_version":__import__("env.tasks.srairl_interaction",fromlist=["REFERENCE_FRAME_VERSION"]).REFERENCE_FRAME_VERSION,
            "road_geometry": __import__("env.tasks.srairl_interaction",fromlist=["ROAD_GEOMETRY_CONFIG"]).ROAD_GEOMETRY_CONFIG if self.dynamics_mode=="metadrive_physics" else None,
            "srairl": self.config.__dict__,
            "training": self.train_config.__dict__, "episode": self.episode, "global_steps": self.global_steps,
            "scene_source":"procedural" if self.dynamics_mode=="metadrive_physics" else "legacy",
            "training_scenarios":sorted({getattr(env,"scenario_type","unknown") for env in self.envs}),
            "dynamics_mode":self.dynamics_mode,
            "reactive_background":bool(getattr(getattr(getattr(self.envs[0].world,"engine",None),"traffic_manager",None),"reactive",False)) if self.envs else False}, indent=2))

    def load_training_state(self, directory):
        """Resume a trusted local checkpoint at an episode boundary."""
        directory = Path(directory)
        metadata = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if metadata.get("srairl") != self.config.__dict__:
            raise ValueError("checkpoint SR-AIRL configuration does not match the current run")
        if metadata.get("training") != self.train_config.__dict__:
            raise ValueError("checkpoint training configuration does not match the current run")
        dynamics = metadata.get("dynamics_mode")
        if dynamics != self.dynamics_mode:
            raise ValueError("checkpoint dynamics mode does not match the current environment")
        if self.dynamics_mode=='metadrive_physics':
            from env.tasks.srairl_interaction import ROAD_GEOMETRY_CONFIG, REFERENCE_FRAME_VERSION
            if metadata.get('road_geometry')!=ROAD_GEOMETRY_CONFIG or metadata.get('reference_frame_version')!=REFERENCE_FRAME_VERSION:
                raise ValueError('checkpoint road/reference frame requires recollection and retraining')
        expected_revision=getattr(self,'source_dataset_revision',None)
        if expected_revision and expected_revision!=metadata.get('source_dataset_revision'):
            raise ValueError('checkpoint source dataset does not match the current run')
        self.source_dataset_revision=metadata.get('source_dataset_revision')
        self.policy_elcv.load(directory / "policy_elcv.pt")
        self.policy_rfv.load(directory / "policy_rfv.pt")
        self.discriminator_elcv.load_state_dict(torch.load(
            directory / "discriminator_elcv.pt", map_location=self.device, weights_only=True))
        self.discriminator_rfv.load_state_dict(torch.load(
            directory / "discriminator_rfv.pt", map_location=self.device, weights_only=True))
        state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
        if int(state.get("seed", -1)) != self.train_config.seed:
            raise ValueError("checkpoint RNG seed does not match the current run")
        self.policy_elcv.optimizer.load_state_dict(state["policy_elcv_optimizer"])
        self.policy_rfv.optimizer.load_state_dict(state["policy_rfv_optimizer"])
        self.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        self.episode = int(state["episode"])
        self.global_steps = int(state["global_steps"])
        self.next_decay_step = int(state["next_decay_step"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return self

    @staticmethod
    def load_policies(directory, config, training_config, device="cpu"):
        """Restore both actors for evaluation without constructing a discriminator."""
        state_dim = config.frame_state_dim * config.history_len
        policies = [PPO(state_dim, 2, training_config.actor_lr, training_config.critic_lr,
                        config.gamma, training_config.ppo_epochs, training_config.clip_epsilon,
                        training_config.initial_action_std, device) for _ in range(2)]
        policies[0].load(Path(directory) / "policy_elcv.pt")
        policies[1].load(Path(directory) / "policy_rfv.pt")
        for policy in policies:
            policy.policy.eval()
            policy.policy_old.eval()
        return tuple(policies)

    def train(self, callback=None):
        history = []
        active_env = None
        for episode in range(self.episode, self.train_config.episodes):
            if self.env_factory is None:
                env=self.envs[episode % len(self.envs)]
                if active_env is not None and active_env is not env and self.dynamics_mode=='metadrive_physics':
                    active_env.close()
                active_env=env
                tracking_return = self._collect_episode(env)
            else:
                env=self.env_factory(episode)
                if not isinstance(env,SRAIRLEnv):
                    raise TypeError("SR-AIRL environment factory must return an SRAIRLEnv")
                if env.dynamics_mode!=self.dynamics_mode:
                    env.close()
                    raise ValueError("SR-AIRL environment factory returned a different dynamics mode")
                self.last_env=env
                try:
                    tracking_return = self._collect_episode(env)
                finally:
                    env.close()
            metrics = {"episode": episode, "global_steps": self.global_steps, "tracking_return": tracking_return}
            sample=getattr(env,'current_initial_state',None)
            if sample:
                metrics.update({f"initial_state_{key}":value for key,value in sample.items()
                                if isinstance(value,(int,float,np.number)) and not isinstance(value,bool)})
                metrics['initial_state_dataset_id']=sample.get('dataset_id')
                metrics['initial_state_recording_group']=sample.get('recording_group')
            if episode % self.train_config.discriminator_every_episodes == 0:
                metrics.update(self._discriminator_update())
                metrics.update({f"elcv_{k}": v for k,v in self.policy_elcv.update().items()})
                metrics.update({f"rfv_{k}": v for k,v in self.policy_rfv.update().items()})
            self.episode = episode + 1
            history.append(metrics)
            if callback:
                keep_going = callback(metrics)
            else:
                keep_going = True
            if self.output_dir and self.episode % self.train_config.discriminator_every_episodes == 0:
                with (self.output_dir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics, allow_nan=False) + "\n")
                self.save()
            if keep_going is False:
                break
        if self.output_dir:
            self.save()
        return history
