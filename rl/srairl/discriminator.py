"""Social AIRL reward and discriminator, following the public upstream airl.py."""
import torch
from torch import nn
import torch.nn.functional as F

from rl.srairl.reward_config import (
    ACC_SCALE, DT, JERK_SCALE, POS_LAT_SCALE, POS_LONG_SCALE, PRED_HORIZON,
    PRED_STEPS, VEL_MIN, VEL_SCALE,
)


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


def extract_private_state(states, role, history_len=11):
    frames = states.reshape(-1, history_len, 16)
    elcv = frames[..., 0:4]
    rfv_rel = frames[..., 4:8]
    if role == "lcv":
        ego = elcv
    elif role == "fv":
        ego = elcv - rfv_rel
    else:
        raise ValueError(role)
    return ego


def extract_motion_features(states, role, history_len=11, *, dt=DT, acceleration_scale=ACC_SCALE,
                            jerk_scale=JERK_SCALE, pos_lat_scale=POS_LAT_SCALE,
                            pos_long_scale=POS_LONG_SCALE, vel_scale=VEL_SCALE, vel_min=VEL_MIN):
    """Build upstream's 8-channel position/velocity/acceleration/jerk tensor."""
    if role not in ("lcv", "fv"):
        raise ValueError(f"Unsupported role: {role}")
    frames = states.reshape(-1, history_len, 16)
    lcv, fv_rel = frames[..., 0:4], frames[..., 4:8]
    ego = lcv if role == "lcv" else lcv - fv_rel
    position = torch.stack((ego[..., 0] * pos_lat_scale,
                            ego[..., 1] * pos_long_scale), dim=-1)
    velocity = ego[..., 2:4] * vel_scale + vel_min
    relative_position = (position - position[:, :1]) / position.new_tensor(
        (pos_lat_scale, pos_long_scale))
    acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1]) / float(dt)
    jerk = torch.diff(acceleration, dim=1, prepend=acceleration[:, :1]) / float(dt)
    features = torch.cat((relative_position, velocity / float(vel_scale),
                          acceleration / float(acceleration_scale),
                          jerk / float(jerk_scale)), dim=-1)
    return features.transpose(1, 2)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, values):
        result = self.conv(values)
        return result[..., :-self.pad] if self.pad else result


class TemporalPrivateReward(nn.Module):
    """Causal three-layer TCN over eight physical motion-feature channels."""
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(input_dim, hidden_dim, dilation=1), nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, dilation=2), nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, dilation=4), nn.ReLU(),
        )
        self.reward_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, 1))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] != 8:
            raise ValueError(f"TCN expects [batch,8,time], got {tuple(features.shape)}")
        return self.reward_head(self.encoder(features)[:, :, -1])


class AnisotropicInteractionPotential(nn.Module):
    """Shared explicit interaction potential from upstream SocialAIRL."""
    def __init__(self, L_long, L_lat, dir_lambda, *, pos_lat_scale=25.0, pos_long_scale=500.0,
                 vel_scale=80.0, vel_offset=-40.0, prediction_horizon=PRED_HORIZON,
                 prediction_steps=PRED_STEPS, prediction_dt=DT):
        super().__init__()
        if min(float(L_long), float(L_lat)) <= 0 or not 0 <= float(dir_lambda) <= 1:
            raise ValueError("invalid interaction-potential parameters")
        self._l_long_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(L_long)))) )
        self._l_lat_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(L_lat)))) )
        bounded = torch.tensor(float(dir_lambda)).clamp(1e-4, 1.0 - 1e-4)
        self._dir_lambda_raw = nn.Parameter(torch.logit(bounded))
        if min(float(pos_lat_scale),float(pos_long_scale),float(vel_scale))<=0:
            raise ValueError("invalid observation normalization scales")
        self.pos_lat_scale=float(pos_lat_scale);self.pos_long_scale=float(pos_long_scale)
        self.vel_scale=float(vel_scale);self.vel_offset=float(vel_offset)
        if float(prediction_horizon) <= 0 or int(prediction_steps) < 1 or float(prediction_dt) <= 0:
            raise ValueError("interaction prediction horizon and sample count must be positive")
        self.prediction_horizon=float(prediction_horizon)
        self.prediction_steps=int(prediction_steps)
        self.prediction_dt=float(prediction_dt)

    @property
    def L_long(self):
        return F.softplus(self._l_long_raw).clamp_min(1e-3)

    @property
    def L_lat(self):
        return F.softplus(self._l_lat_raw).clamp_min(1e-3)

    @property
    def dir_lambda(self):
        return torch.sigmoid(self._dir_lambda_raw)

    def _physical_state(self,x):
        lat = x[:, 0] * self.pos_lat_scale
        lon = x[:, 1] * self.pos_long_scale
        v_long = x[:, 2] * self.vel_scale + self.vel_offset
        v_lat = x[:, 3] * self.vel_scale + self.vel_offset
        return lat, lon, v_long, v_lat

    def pair_potential(self, target, neighbor):
        lat_i, lon_i, v_long_i, v_lat_i = self._physical_state(target)
        lat_j, lon_j, v_long_j, v_lat_j = self._physical_state(neighbor)
        d_long, d_lat = lon_j - lon_i, lat_j - lat_i
        rel_long, rel_lat = v_long_j - v_long_i, v_lat_j - v_lat_i
        psi = torch.atan2(v_lat_i, v_long_i)
        cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)
        x0 = cos_psi * d_long + sin_psi * d_lat
        y0 = -sin_psi * d_long + cos_psi * d_lat
        vx = cos_psi * rel_long + sin_psi * rel_lat
        vy = -sin_psi * rel_long + cos_psi * rel_lat
        horizon = torch.linspace(self.prediction_dt, self.prediction_horizon, self.prediction_steps,
                                 device=target.device, dtype=target.dtype)
        x_pred = x0[:, None] + vx[:, None] * horizon
        y_pred = y0[:, None] + vy[:, None] * horizon
        q = (x_pred / self.L_long).square() + (y_pred / self.L_lat).square()
        spatial = torch.exp(-q)
        distance = torch.linalg.vector_norm(torch.stack((x_pred,y_pred),dim=-1),dim=-1)
        cos_phi = torch.where(distance > 1e-4,
                              x_pred / distance.clamp_min(1e-4),
                              torch.ones_like(distance))
        direction_weight = self.dir_lambda + (1.0 - self.dir_lambda) * (1.0 + cos_phi) / 2.0
        return (direction_weight * spatial).mean(dim=1)

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
                 hidden_units_r=(64, 64), hidden_units_v=(64, 64), role="lcv", history_len=None,
                 dt=DT, acceleration_scale=ACC_SCALE, jerk_scale=JERK_SCALE):
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
        self.motion_feature_config={
            "dt":float(dt),"acceleration_scale":float(acceleration_scale),
            "jerk_scale":float(jerk_scale),"pos_lat_scale":shared_interaction_potential.pos_lat_scale,
            "pos_long_scale":shared_interaction_potential.pos_long_scale,
            "vel_scale":shared_interaction_potential.vel_scale,"vel_min":shared_interaction_potential.vel_offset,
        }
        self.private_g = TemporalPrivateReward()
        self.h = build_mlp(state_shape, 1, hidden_units_v, nn.ReLU(inplace=True))
        alpha = torch.tensor(float(alpha_init))
        self.alpha_raw = nn.Parameter(torch.log(torch.expm1(alpha)))

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def get_reward(self, states):
        motion_features = extract_motion_features(states, self.role, self.history_len,
                                                   **self.motion_feature_config)
        r_private = self.private_g(motion_features)
        phi = self.interaction_potential(states, role=self.role)
        return r_private - self.alpha * phi, r_private, phi

    def f(self, states, dones, next_states):
        reward, _, _ = self.get_reward(states)
        done = dones.reshape(-1, 1).float()
        return reward + self.gamma * (1.0 - done) * self.h(next_states) - self.h(states)

    def discriminator_logits(self, states, dones, log_pis, next_states):
        return self.f(states, dones, next_states) - log_pis.reshape(-1, 1)

    def forward(self, states, dones, log_pis, next_states):
        return torch.sigmoid(self.discriminator_logits(states, dones, log_pis, next_states))

    def calculate_policy_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            return self.discriminator_logits(states, dones, log_pis, next_states)

    # Retain the public method name used by older local callers.
    calculate_reward = calculate_policy_reward
