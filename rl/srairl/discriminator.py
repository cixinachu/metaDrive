"""Social AIRL reward and discriminator, ported from the public upstream code.

Math follows Step1_Reward_Recovery/airl.py at the commit pinned in UPSTREAM.md.
The configurable constants come from the missing upstream reward_config.py.
"""
import torch
from torch import nn
import torch.nn.functional as F


def build_mlp(input_dim, output_dim, hidden_units=(64, 64), hidden_activation=None, output_activation=None):
    activation = hidden_activation if hidden_activation is not None else nn.Tanh()
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.extend((nn.Linear(units, next_units), activation))
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def split_latest_state(states):
    latest = states[:, -16:]
    elcv = latest[:, 0:4]
    rfv_rel = latest[:, 4:8]
    return {
        "lcv": elcv,
        "fv": elcv - rfv_rel,
        "nlv": elcv + latest[:, 8:12],
        "olv": elcv + latest[:, 12:16],
    }


def extract_private_state(states, role, history_len=4):
    frames = states.reshape(-1, history_len, 16)
    elcv = frames[..., 0:4]
    rfv_rel = frames[..., 4:8]
    if role == "lcv":
        ego = elcv
    elif role == "fv":
        ego = elcv - rfv_rel
    else:
        raise ValueError(role)
    return ego.reshape(states.shape[0], -1)


class AnisotropicInteractionPotential(nn.Module):
    """Shared explicit interaction potential from upstream SocialAIRL."""
    def __init__(self, L_long, L_lat, dir_lambda, *, pos_lat_scale=25.0, pos_long_scale=500.0,
                 vel_scale=80.0, vel_offset=-40.0):
        super().__init__()
        if min(float(L_long), float(L_lat)) <= 0 or not 0 <= float(dir_lambda) <= 1:
            raise ValueError("invalid interaction-potential parameters")
        self.L_long = float(L_long)
        self.L_lat = float(L_lat)
        self.dir_lambda = float(dir_lambda)
        if min(float(pos_lat_scale),float(pos_long_scale),float(vel_scale))<=0:
            raise ValueError("invalid observation normalization scales")
        self.pos_lat_scale=float(pos_lat_scale);self.pos_long_scale=float(pos_long_scale)
        self.vel_scale=float(vel_scale);self.vel_offset=float(vel_offset)

    def _physical_state(self,x):
        lat = x[:, 0] * self.pos_lat_scale
        lon = x[:, 1] * self.pos_long_scale
        v_long = x[:, 2] * self.vel_scale + self.vel_offset
        v_lat = x[:, 3] * self.vel_scale + self.vel_offset
        return lat, lon, v_long, v_lat

    def pair_potential(self, target, neighbor):
        lat_i, lon_i, v_long_i, v_lat_i = self._physical_state(target)
        lat_j, lon_j, _, _ = self._physical_state(neighbor)
        d_long, d_lat = lon_j - lon_i, lat_j - lat_i
        psi = torch.atan2(v_lat_i, v_long_i)
        cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)
        dx_e = cos_psi * d_long + sin_psi * d_lat
        dy_e = -sin_psi * d_long + cos_psi * d_lat
        q = (dx_e / self.L_long) ** 2 + (dy_e / self.L_lat) ** 2
        spatial = torch.exp(-0.5 * q)
        phi = torch.atan2(dy_e, dx_e)
        direction_weight = self.dir_lambda + (1.0 - self.dir_lambda) * (1.0 + torch.cos(phi)) / 2.0
        return direction_weight * spatial

    def forward(self, states, role="lcv"):
        vehicles = split_latest_state(states)
        if role == "lcv":
            target = vehicles["lcv"]
            neighbors = [vehicles["fv"], vehicles["nlv"], vehicles["olv"]]
        elif role == "fv":
            target = vehicles["fv"]
            neighbors = [vehicles["lcv"], vehicles["nlv"], vehicles["olv"]]
        else:
            raise ValueError(f"Unsupported role: {role}")
        return torch.stack([self.pair_potential(target, n) for n in neighbors], dim=-1).sum(dim=-1, keepdim=True)


class SocialAIRLDiscrim(nn.Module):
    """Two-actor AIRL discriminator using upstream r_private - alpha*phi."""
    def __init__(self, state_shape, gamma, shared_interaction_potential, alpha_init,
                 hidden_units_r=(64, 64), hidden_units_v=(64, 64), role="lcv", history_len=None):
        super().__init__()
        if role not in ("lcv", "fv"):
            raise ValueError(f"Unsupported role: {role}")
        if history_len is None:
            history_len = state_shape // 16
        if history_len < 1 or state_shape != history_len * 16:
            raise ValueError("state_shape must equal history_len * 16")
        if alpha_init <= 0 or not 0 < gamma <= 1:
            raise ValueError("alpha_init and gamma must be positive (gamma <= 1)")
        self.role, self.history_len, self.gamma = role, int(history_len), float(gamma)
        self.interaction_potential = shared_interaction_potential
        self.private_g = build_mlp(4 * history_len, 1, hidden_units_r, nn.ReLU(inplace=True))
        self.h = build_mlp(state_shape, 1, hidden_units_v, nn.ReLU(inplace=True))
        alpha = torch.tensor(float(alpha_init))
        self.alpha_raw = nn.Parameter(torch.log(torch.expm1(alpha)))

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def get_reward(self, states):
        private_state = extract_private_state(states, self.role, self.history_len)
        r_private = self.private_g(private_state)
        phi = self.interaction_potential(states, role=self.role)
        return r_private - self.alpha * phi, r_private, phi

    def f(self, states, dones, next_states):
        reward, _, _ = self.get_reward(states)
        done = dones.reshape(-1, 1).float()
        return reward + self.gamma * (1.0 - done) * self.h(next_states) - self.h(states)

    def forward(self, states, dones, log_pis, next_states):
        logits = torch.clamp(self.f(states, dones, next_states) - log_pis.unsqueeze(-1), -50.0, 50.0)
        return torch.sigmoid(logits)

    def calculate_reward(self, states, dones=None, log_pis=None, next_states=None):
        with torch.no_grad():
            return self.get_reward(states)[0]
