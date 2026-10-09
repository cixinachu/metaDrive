"""SAC-Lagrangian, Gaussian WCSAC and WCSAC-IQN on the SB3 SAC backbone.

Cost critics predict discounted *remaining* cost at replay states. The budget is
in those units, not undiscounted episode cost or seconds. See docs/STUDIO.md.
"""
from copy import deepcopy
from typing import NamedTuple
import math
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update


class CostSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    costs: th.Tensor


class CostReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        costs = np.asarray([i['cost'] for i in infos], dtype=np.float32)
        if not np.isfinite(costs).all() or np.any((costs < 0) | (costs > 1)):
            raise ValueError('Expected finite environment cost in [0,1]')
        self.costs[self.pos] = costs
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        idx = np.random.randint(self.n_envs, size=len(batch_inds))
        next_obs = self.observations[(batch_inds+1) % self.buffer_size, idx] if self.optimize_memory_usage else self.next_observations[batch_inds, idx]
        data = (self._normalize_obs(self.observations[batch_inds, idx], env), self.actions[batch_inds, idx],
                self._normalize_obs(next_obs, env),
                (self.dones[batch_inds, idx]*(1-self.timeouts[batch_inds, idx])).reshape(-1, 1),
                self._normalize_reward(self.rewards[batch_inds, idx].reshape(-1, 1), env),
                self.costs[batch_inds, idx].reshape(-1, 1))
        return CostSamples(*map(self.to_torch, data))


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, outputs))


class SafetyCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, kind, width=256, cosine_dim=64):
        super().__init__()
        self.kind = kind
        if kind == 'iqn':
            self.features = mlp(obs_dim+action_dim, width, width)
            self.embedding = nn.Linear(cosine_dim, width)
            self.output = nn.Sequential(nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
            self.register_buffer('frequencies', th.arange(1, cosine_dim+1).float()*math.pi)
        else:
            self.mean = mlp(obs_dim+action_dim, 1, width)
            if kind == 'gaussian':
                self.variance = mlp(obs_dim+action_dim, 1, width)

    def forward(self, obs, actions, taus=None):
        x = th.cat([obs.flatten(1), actions.flatten(1)], dim=1)
        if self.kind == 'iqn':
            if taus is None:
                raise ValueError('IQN requires quantile fractions')
            features = self.features(x).unsqueeze(1)
            embedding = F.relu(self.embedding(th.cos(taus.unsqueeze(-1)*self.frequencies)))
            return F.softplus(self.output(features*embedding).squeeze(-1))
        mean = F.softplus(self.mean(x))
        if self.kind == 'gaussian':
            return mean, F.softplus(self.variance(x))+1e-6
        return mean


def gaussian_cvar(mean, variance, tail_fraction):
    if tail_fraction == 1:
        return mean
    normal = th.distributions.Normal(th.tensor(0., device=mean.device), th.tensor(1., device=mean.device))
    z = normal.icdf(th.tensor(1-tail_fraction, device=mean.device))
    return mean + th.exp(normal.log_prob(z))/tail_fraction * th.sqrt(variance.clamp_min(1e-8))


def quantile_huber_loss(prediction, target, taus, kappa=1.):
    delta = target.unsqueeze(1) - prediction.unsqueeze(2)
    absolute = delta.abs()
    huber = th.where(absolute <= kappa, .5*delta.square(), kappa*(absolute-.5*kappa))
    weights = (taus.unsqueeze(2) - (delta.detach() < 0).float()).abs()
    return (weights*huber/kappa).mean()


def gaussian_targets(costs, discount, next_mean, next_variance, current_mean):
    mean = costs + discount*next_mean
    variance = (costs.square() + 2*discount*costs*next_mean +
                discount.square()*(next_variance+next_mean.square()) - current_mean.square()).clamp_min(1e-6)
    # An absorbing transition has deterministic immediate cost, independent of approximation error.
    variance = th.where(discount == 0, th.full_like(variance, 1e-6), variance)
    return mean, variance


class ConstrainedSAC(SAC):
    safety_kind = 'mean'

    def __init__(self, *args, cost_limit=5., gamma_cost=.99, tail_fraction=.1,
                 safety_lr=3e-4, dual_lr=1e-4, initial_multiplier=.1,
                 n_quantiles=32, safety_width=256, **kwargs):
        if not 0 < tail_fraction <= 1 or not 0 <= gamma_cost < 1:
            raise ValueError('Invalid cost discount / CVaR tail fraction')
        if cost_limit < 0 or initial_multiplier <= 0 or n_quantiles < 2:
            raise ValueError('Invalid cost budget / multiplier / quantile count')
        self.cost_limit, self.gamma_cost, self.tail_fraction = cost_limit, gamma_cost, tail_fraction
        self.safety_lr, self.dual_lr = safety_lr, dual_lr
        self.initial_multiplier, self.n_quantiles, self.safety_width = initial_multiplier, n_quantiles, safety_width
        kwargs['replay_buffer_class'] = CostReplayBuffer
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        self.safety_critic = SafetyCritic(int(np.prod(self.observation_space.shape)), int(np.prod(self.action_space.shape)), self.safety_kind, self.safety_width).to(self.device)
        self.safety_target = deepcopy(self.safety_critic).to(self.device).requires_grad_(False)
        self.safety_optimizer = th.optim.Adam(self.safety_critic.parameters(), lr=self.safety_lr)
        self.log_multiplier = th.tensor([math.log(math.expm1(self.initial_multiplier))], device=self.device, requires_grad=True)
        self.dual_optimizer = th.optim.Adam([self.log_multiplier], lr=self.dual_lr)

    def _get_torch_save_params(self):
        states, variables = super()._get_torch_save_params()
        return states+['safety_critic', 'safety_target', 'safety_optimizer', 'dual_optimizer'], variables+['log_multiplier']

    def risk(self, obs, actions):
        if self.safety_kind == 'iqn':
            taus = 1-self.tail_fraction + self.tail_fraction*th.rand((len(obs), self.n_quantiles), device=self.device)
            return self.safety_critic(obs, actions, taus).mean(1, keepdim=True)
        out = self.safety_critic(obs, actions)
        return gaussian_cvar(*out, self.tail_fraction) if self.safety_kind == 'gaussian' else out

    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        records = []
        for step in range(gradient_steps):
            b = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()
            actions, logp = self.actor.action_log_prob(b.observations)
            logp = logp.reshape(-1, 1)
            ent_loss = th.zeros((), device=self.device)
            if self.ent_coef_optimizer:
                entropy = self.log_ent_coef.detach().exp()
                ent_loss = -(self.log_ent_coef*(logp+self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad(); ent_loss.backward(); self.ent_coef_optimizer.step()
            else:
                entropy = self.ent_coef_tensor
            with th.no_grad():
                next_actions, next_logp = self.actor.action_log_prob(b.next_observations)
                next_q = th.cat(self.critic_target(b.next_observations, next_actions), dim=1).min(1, keepdim=True).values
                target_q = b.rewards+(1-b.dones)*self.gamma*(next_q-entropy*next_logp.reshape(-1, 1))
                discount = (1-b.dones)*self.gamma_cost
            q_loss = .5*sum(F.mse_loss(q, target_q) for q in self.critic(b.observations, b.actions))
            self.critic.optimizer.zero_grad(); q_loss.backward(); self.critic.optimizer.step()
            if self.safety_kind == 'iqn':
                taus = th.rand((batch_size, self.n_quantiles), device=self.device)
                with th.no_grad():
                    next_taus = th.rand_like(taus)
                    target = b.costs+discount*self.safety_target(b.next_observations, next_actions, next_taus)
                safety_loss = quantile_huber_loss(self.safety_critic(b.observations, b.actions, taus), target, taus)
            elif self.safety_kind == 'gaussian':
                mean, variance = self.safety_critic(b.observations, b.actions)
                with th.no_grad():
                    nm, nv = self.safety_target(b.next_observations, next_actions)
                    tm, tv = gaussian_targets(b.costs, discount, nm, nv, mean.detach())
                safety_loss = F.mse_loss(mean, tm)+F.mse_loss(variance.sqrt(), tv.sqrt())
            else:
                with th.no_grad():
                    target = b.costs+discount*self.safety_target(b.next_observations, next_actions)
                safety_loss = F.mse_loss(self.safety_critic(b.observations, b.actions), target)
            self.safety_optimizer.zero_grad(); safety_loss.backward(); self.safety_optimizer.step()
            # Freeze critic weights for actor update, retaining d risk / d action.
            self.safety_critic.requires_grad_(False)
            self.critic.requires_grad_(False)
            risk = self.risk(b.observations, actions)
            q = th.cat(self.critic(b.observations, actions), dim=1).min(1, keepdim=True).values
            multiplier = F.softplus(self.log_multiplier)
            actor_loss = (entropy*logp-q+multiplier.detach()*risk).mean()
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            self.safety_critic.requires_grad_(True)
            self.critic.requires_grad_(True)
            violation = (risk.detach()-self.cost_limit).mean()
            dual_loss = -multiplier*violation
            self.dual_optimizer.zero_grad(); dual_loss.backward(); self.dual_optimizer.step()
            with th.no_grad():
                self.log_multiplier.clamp_(-20, 100)
            if (self._n_updates+step) % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.safety_critic.parameters(), self.safety_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.)
            record = dict(actor_loss=actor_loss.item(), critic_loss=q_loss.item(), safety_loss=safety_loss.item(),
                          ent_coef=entropy.item(), ent_coef_loss=ent_loss.item(), multiplier=multiplier.item(),
                          predicted_risk=risk.mean().item(), constraint_violation=violation.item(), dual_loss=dual_loss.item())
            if not all(math.isfinite(v) for v in record.values()):
                raise FloatingPointError(f'Nonfinite optimization metric: {record}')
            records.append(record)
        self._n_updates += gradient_steps
        for name in records[0] if records else []:
            self.logger.record('train/'+name, float(np.mean([r[name] for r in records])))
        self.logger.record('train/n_updates', self._n_updates)
        self.logger.record('train/cost_limit', self.cost_limit)


# Preserve old imports and checkpoint pickle references after the module split.
def __getattr__(name):
    from importlib import import_module
    modules = {'SACLagrangian': 'rl.sac_lagrangian', 'WCSAC': 'rl.wcsac', 'WCSACIQN': 'rl.wcsac_iqn'}
    if name in modules:
        return getattr(import_module(modules[name]), name)
    if name == 'ALGORITHMS':
        from rl.algorithms import ALGORITHMS
        return ALGORITHMS
    raise AttributeError(name)
