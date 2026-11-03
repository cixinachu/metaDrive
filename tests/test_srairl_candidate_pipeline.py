import numpy as np
import pytest
import torch

from env.tasks.srairl_trajectory_tracker import SRAIRLTrajectoryTracker
from rl.srairl.pipeline.coordinates import lane_state_to_upstream, upstream_to_lane_state
from rl.srairl.pipeline.executor import RoleHistoryBuffer, SRAIRLClosedLoopPipeline
from rl.srairl.pipeline.model_bundle import SRAIRLModelBundle
from rl.srairl.pipeline.normalization import (
    PlannerGoalNormalizer, RewardPredictorNormalizer, reconstruct_predictor_deltas,
)
from rl.srairl.pipeline.schema import JointCandidate, TrajectoryCandidate
from rl.srairl.pipeline.selector import CandidateSelector


class StraightLane:
    def local_coordinates(self, point):
        return float(point[0]), float(point[1])

    def position(self, longitudinal, lateral=0):
        return np.array([longitudinal, lateral, 0.0])

    def heading_theta_at(self, longitudinal):
        return 0.0


def trajectory(y=0.0, x0=0.0, *, actions=True):
    states = np.column_stack((np.full(100, y), x0 + np.arange(100),
                              np.full(100, 10.0), np.zeros(100)))
    acts = np.zeros((100, 2), dtype=float) if actions else None
    return TrajectoryCandidate(states, acts, 0.1)


class LinearReward(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.feature = feature
        self.history_len = 4

    def get_reward(self, states):
        return states[:, self.feature:self.feature + 1] + self.anchor * 0, None, None


def joint(candidate_id, elcv_y, elcv_actions=None):
    return JointCandidate(
        candidate_id,
        TrajectoryCandidate(trajectory(elcv_y).states,
                            np.zeros((100, 2)) if elcv_actions is None else elcv_actions, 0.1),
        trajectory(0, 12), trajectory(0, 24, actions=False), trajectory(0, 36, actions=False),
    )


def test_candidate_schema_enforces_upstream_state_and_action_contract():
    item = trajectory()
    assert item.states.shape == (100, 4)
    assert item.actions.shape == (100, 2)
    with pytest.raises(ValueError, match=r"\[T,4\]"):
        TrajectoryCandidate(np.zeros((100, 6)), dt=0.1)
    with pytest.raises(ValueError, match="finite"):
        TrajectoryCandidate(np.full((100, 4), np.nan), dt=0.1)


def test_coordinate_round_trip_matches_tracker_lateral_sign():
    source = np.array([[2.0, 30.0, 12.0, -0.5], [3.5, 40.0, 9.0, 0.2]])
    lane = StraightLane()
    lane_states = upstream_to_lane_state(source, lane)
    assert np.allclose(lane_states[:, 0], source[:, 1])
    assert np.allclose(lane_states[:, 1], -source[:, 0])
    recovered = lane_state_to_upstream(lane_states, lane)
    assert np.allclose(source, recovered, atol=1e-8)


def test_training_normalizers_and_four_dimensional_delta_reconstruction():
    raw = np.array([[25.0, 250.0, 0.0, 40.0]])
    assert np.allclose(PlannerGoalNormalizer.transform(raw), [[0.5, 0.5, 0.5, 1.0]])
    assert np.allclose(RewardPredictorNormalizer.transform(raw), [[1.0, 0.5, 0.5, 1.0]])
    history = np.array([[1.0, 10.0, 5.0, 0.0], [1.0, 10.5, 5.0, 0.0]])
    deltas = np.tile([0.0, 0.5, 0.2, 0.0], (3, 1))
    future, acceleration = reconstruct_predictor_deltas(history, deltas, dt=0.1)
    assert future.shape == (3, 4)
    assert np.allclose(future[-1], [1.0, 12.0, 5.6, 0.0])
    assert np.allclose(acceleration[:, 0], 2.0)


def test_reward_guide_gradient_is_per_step_and_depends_on_predicted_fv():
    from rl.srairl.upstream.Step4_Predictor.model.ddpm import GaussianDiffusionSampler

    class ZeroNoise(torch.nn.Module):
        def forward(self, t, x, origin, context, mask):
            return torch.zeros_like(x)

    class Guide(torch.nn.Module):
        def forward(self, state, t):
            return state[..., 4:5]

    sampler = GaussianDiffusionSampler(ZeroNoise(), beta_1=1e-4, beta_T=.05, T=100)
    state = {
        "rfv_history_last": torch.tensor([[0.0, 100.0, 10.0, 0.0]]),
        "n_lcv": torch.tensor([[[0.5, 0.2, 0.6, 0.5]] * 100]),
        "n_nlv": torch.tensor([[[0.5, 0.3, 0.6, 0.5]] * 100]),
        "n_olv": torch.tensor([[[0.5, 0.1, 0.6, 0.5]] * 100]),
    }
    deltas = torch.zeros((1, 100, 4))
    gradients = sampler.guide_fn(deltas, torch.tensor([10]), Guide(), state, guide_scale=1.0)
    assert gradients.shape == deltas.shape
    assert torch.isfinite(gradients).all()
    assert gradients[..., 0].abs().sum() > 0
    assert gradients[0, 0, 0].abs() > gradients[0, -1, 0].abs() > 0


def test_model_bundle_rejects_dimension_and_revision_mismatch(tmp_path):
    paths = []
    for name in ("goal", "planner", "guide", "predictor", "reward"):
        path = tmp_path / name
        path.write_bytes(b"checkpoint")
        paths.append(path)
    with pytest.raises(ValueError, match="dataset revisions"):
        SRAIRLModelBundle(*paths,
            planner_metadata={"horizon": 100, "dataset_revision": "a"},
            predictor_metadata={"horizon": 100, "dataset_revision": "b"},
            normalization={"planner_goal": "planner-goal-v1",
                           "reward_predictor": "reward-predictor-v1"})


def test_selector_uses_srairl_rewards_and_rejects_uncalibrated_actions():
    selector = CandidateSelector(LinearReward(48), LinearReward(48),
                                 validated_ax=(-1.5, 1.5), validated_ay=(-1.5, 1.5))
    first = joint(0, 1.0)
    second = joint(1, 2.0)
    rejected = joint(2, 4.0, np.tile([3.0, 0.0], (100, 1)))
    selected, diagnostics = selector.select([first, second, rejected])
    assert selected.candidate_id == 1
    assert rejected.feasibility["feasible"] is False
    assert any("outside_calibration" in reason for reason in rejected.feasibility["reasons"])
    assert len(diagnostics) == 3


def test_history_buffer_pads_initial_frame_and_requires_four_roles():
    roles = {name: np.array([0.0, 1.0, 2.0, 0.0]) for name in
             ("elcv", "rfv", "rlv_current", "rlv_target")}
    buffer = RoleHistoryBuffer(20)
    buffer.reset(roles)
    assert all(value.shape == (20, 4) for value in buffer.current().values())
    with pytest.raises(ValueError, match="four explicit"):
        buffer.append({"elcv": roles["elcv"]})


def test_one_closed_loop_episode_executes_dual_tracker_and_replans():
    lane = StraightLane()

    class Vehicle:
        def __init__(self, s):
            self.position = np.array([s, 0.0, 0.0])
            self.velocity = np.array([10.0, 0.0, 0.0])
            self.lane = lane

    class World:
        def __init__(self):
            self.vehicles = {"elcv": Vehicle(0.0), "rfv": Vehicle(-12.0)}

        def role_vehicle(self, role):
            return self.vehicles[role]

    class Env:
        max_steps = 3

        def __init__(self):
            self.world = World()
            self.steps = 0
            self.scenario = type("Scenario", (), {"scenario_id": "test"})()

        def reset(self, seed=None):
            self.steps = 0
            return {}, {}

        def _roles_live(self):
            return {"elcv": np.array([0., 10. * self.steps, 10., 0.]),
                    "rfv": np.array([0., -12. + 10. * self.steps, 10., 0.]),
                    "rlv_current": np.array([0., 24. + 10. * self.steps, 10., 0.]),
                    "rlv_target": np.array([0., 36. + 10. * self.steps, 10., 0.])}

        def step(self, actions):
            assert set(actions) == {"elcv", "rfv"}
            self.steps += 1
            for vehicle in self.world.vehicles.values():
                vehicle.position[0] += 1.0
            return {}, {"terminated": self.steps >= 3, "done": self.steps >= 3,
                        "lane_change_success": True, "minimum_ttc_s": 2.0,
                        "minimum_distance_m": 8.0, "route_completion": self.steps / 3}

    class Planner:
        def generate_candidates(self, history, num_candidates, seed):
            assert all(value.shape == (20, 4) for value in history.values())
            return [trajectory(0.0, 0.0)]

    class Predictor:
        def predict(self, rfv_history, elcv_future, rlv_current_future, rlv_target_future):
            return trajectory(0.0, -12.0)

    class Forecaster:
        def __call__(self, env, history, horizon, dt):
            return trajectory(0.0, 24.0, actions=False), trajectory(0.0, 36.0, actions=False)

    env = Env()
    pipeline = SRAIRLClosedLoopPipeline(
        planner=Planner(), predictor=Predictor(),
        selector=CandidateSelector(LinearReward(48), LinearReward(48)),
        tracker=SRAIRLTrajectoryTracker(action_adapter=None), role_forecaster=Forecaster(),
        candidate_count=1, replan_steps=2,
        lane_state_mapper=lambda role, state, env: np.array([state[1], -state[0], state[2], state[3]]),
    )
    result = pipeline.evaluate_episode(env, seed=9)
    assert result["metrics"].planning_steps == 2
    assert result["metrics"].selected_candidate_ids == [0, 0]
    assert result["metrics"].lane_change_success
    assert result["last_info"]["done"]
