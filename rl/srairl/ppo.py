"""Continuous PPO used by the upstream SR-AIRL interaction policy.

Network widths, tanh actor, Gaussian action distribution, clipped objective,
return normalization and entropy/value coefficients follow the upstream PPO.py.
The device is explicit so separate tasks do not choose CUDA during import.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards = [], [], [], []
        self.state_values, self.is_terminals, self.next_states = [], [], []

    def add(self, state, action, logprob, value, reward, done, next_state):
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.logprobs.append(logprob.detach())
        self.state_values.append(value.detach())
        self.rewards.append(float(reward))
        self.is_terminals.append(bool(done))
        self.next_states.append(next_state.detach())

    def tensors(self):
        if not self.states:
            raise ValueError("empty PPO rollout")
        return {
            "states": torch.cat(self.states, dim=0),
            "actions": torch.cat(self.actions, dim=0),
            "logprobs": torch.cat(self.logprobs, dim=0),
            "values": torch.cat(self.state_values, dim=0).squeeze(-1),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32, device=self.states[0].device),
            "dones": torch.as_tensor(self.is_terminals, dtype=torch.float32, device=self.states[0].device),
            "next_states": torch.cat(self.next_states, dim=0),
        }

    def sample(self, batch_size):
        data = self.tensors()
        indices = torch.randint(len(self.states), (batch_size,), device=data["states"].device)
        return {key: value[indices] for key, value in data.items()}

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super().__init__()
        self.action_dim = int(action_dim)
        self.register_buffer("action_var", torch.full((action_dim,), float(action_std_init) ** 2, device=device))
        self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                    nn.Linear(64, 1))

    def distribution(self, states):
        mean = self.actor(states)
        covariance = torch.diag(self.action_var).expand(mean.shape[:-1] + (self.action_dim, self.action_dim))
        return MultivariateNormal(mean, covariance)

    def act(self, state):
        dist = self.distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(state)

    def evaluate(self, states, actions):
        dist = self.distribution(states)
        return dist.log_prob(actions), self.critic(states), dist.entropy()

    def set_action_std(self, action_std):
        self.action_var.fill_(float(action_std) ** 2)


class PPO:
    def __init__(self, state_dim, action_dim=2, lr_actor=3e-4, lr_critic=1e-3, gamma=.99,
                 K_epochs=80, eps_clip=.2, action_std_init=.2, device="cpu"):
        self.device = torch.device(device)
        self.gamma, self.eps_clip, self.K_epochs = float(gamma), float(eps_clip), int(K_epochs)
        self.action_std = float(action_std_init)
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init, self.device).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])
        self.mse = nn.MSELoss()

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action, logprob, value = self.policy_old.act(state_t)
        return state_t, action, logprob, value

    def decay_action_std(self, decay_rate=.05, min_action_std=.1):
        self.action_std = max(float(min_action_std), round(self.action_std - float(decay_rate), 4))
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)
        return self.action_std

    def update(self):
        data = self.buffer.tensors()
        returns, discounted = [], 0.0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if done:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        advantages = returns - data["values"].detach()
        losses = []
        for _ in range(self.K_epochs):
            logprobs, values, entropy = self.policy.evaluate(data["states"], data["actions"])
            values = values.squeeze(-1)
            ratios = torch.exp(logprobs - data["logprobs"].detach())
            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.minimum(unclipped, clipped) + .5*self.mse(values, returns) - .01*entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            losses.append(float(loss.mean().detach().cpu()))
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return {"ppo_loss": float(np.mean(losses))}

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "policy_old": self.policy_old.state_dict(),
                    "action_std": self.action_std}, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(state["policy"])
        self.policy_old.load_state_dict(state.get("policy_old", state["policy"]))
        self.action_std = float(state.get("action_std", self.action_std))
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)
