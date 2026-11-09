import numpy as np
import pytest
import torch

from env.tasks.srairl_trajectory_tracker import SRAIRLTrajectoryTracker
from rl.srairl.pipeline.coordinates import lane_state_to_upstream, upstream_to_lane_state
from rl.srairl.pipeline.candidate_builder import build_joint_candidates
from rl.srairl.pipeline.executor import RoleHistoryBuffer, SRAIRLClosedLoopPipeline
from rl.srairl.pipeline.model_bundle import SRAIRLModelBundle
from rl.srairl.pipeline.normalization import (
    PlannerGoalNormalizer, RewardPredictorNormalizer, reconstruct_predictor_deltas,
)
from rl.srairl.pipeline.schema import JointCandidate, PlanningContext, TrajectoryCandidate
from rl.srairl.pipeline.selector import CandidateSelector


class StraightLane:
    length = 10000.0
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


def test_planner_inference_loads_metadata_checkpoints_and_uses_target_lane(tmp_path):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from rl.srairl.pipeline.planner_inference import PlannerInference
    upstream_planner = Path(__file__).resolve().parents[1] / "rl/srairl/upstream/Step2_Planner"
    if str(upstream_planner) not in sys.path:
        sys.path.insert(0, str(upstream_planner))
    from rl.srairl.upstream.Step2_Planner.train_goal import GoalPredictor
    from rl.srairl.upstream.Step2_Planner.model.temporal import TemporalUnet
    from rl.srairl.upstream.Step2_Planner.model.diffusion import GaussianDiffusion

    torch.manual_seed(3)
    goal = GoalPredictor(128)
    goal_path = tmp_path / "goal.pt"
    torch.save({"state_dict": goal.state_dict(), "bound_min": [0., -10.],
                "bound_max": [500., 10.], "architecture": {"context_dim": 128}}, goal_path)
    model = TemporalUnet(horizon=100, transition_dim=6, dim=64, cond_dim=4,
                         dim_mults=(1, 2, 4, 8), attention=False)
    diffusion = GaussianDiffusion(model, horizon=100, observation_dim=4, action_dim=2,
                                  n_timesteps=20, loss_type="l2", clip_denoised=False,
                                  predict_epsilon=False, action_weight=10, loss_discount=1)
    planner_path = tmp_path / "planner.pt"
    torch.save({"ema": diffusion.state_dict()}, planner_path)

    class Bundle:
        planner_goal_path = goal_path
        planner_diffusion_path = planner_path

    inference = PlannerInference(Bundle(), device="cpu")
    history = {role: np.column_stack((np.full(20, y), np.arange(20),
                                      np.full(20, 10.), np.zeros(20))).astype(np.float32)
               for role, y in (("elcv", 0.), ("rfv", 3.5),
                               ("rlv_current", 0.), ("rlv_target", 3.5))}
    # Keep the real checkpoint-backed Goal Predictor, while capturing the hard
    # diffusion conditions so the target-lane terminal is directly asserted.
    observed = {}
    def sample(conditions, verbose=False):
        observed.update({key: value.clone() for key, value in conditions.items()})
        trajectories = torch.zeros((1, 100, 6))
        trajectories[0, :, 3] = torch.arange(100, dtype=torch.float32)
        trajectories[0, :, 4] = 10.0
        return SimpleNamespace(trajectories=trajectories)
    inference.diffusion.conditional_sample = sample
    candidates = inference.generate_candidates(history, PlanningContext(3.5), num_candidates=1, seed=7)
    assert len(candidates) == 1
    assert torch.allclose(observed[0][0], torch.tensor([0.0, 19.0, 10.0, 0.0]))
    assert torch.allclose(observed[99][0, 0], torch.tensor(3.5))
    assert torch.allclose(observed[99][0, 3], torch.tensor(0.0))
    assert candidates[0].states[0, 1] == pytest.approx(1.0)
    assert candidates[0].states[-1, 1] == pytest.approx(100.0)


def test_predictor_inference_loads_checkpoints_and_reconstructs_physical_state(tmp_path):
    from types import SimpleNamespace
    from rl.srairl.pipeline.predictor_inference import RewardGuidedResponsePredictor
    from rl.srairl.upstream.Step3_Reward_Guide.model.backbone import NoisyReward
    from rl.srairl.upstream.Step4_Predictor.model.backbone import TransformerConcatLinear

    torch.manual_seed(11)
    guide = NoisyReward(64, 100)
    model = TransformerConcatLinear(64, 100)
    guide_path, predictor_path = tmp_path / "guide.pt", tmp_path / "predictor.pt"
    checkpoint = {"metadata": {"obs_len": 20, "pred_len": 100},
                  "architecture": {"context_dim": 64, "T": 100}}
    torch.save({**checkpoint, "state_dict": guide.state_dict()}, guide_path)
    torch.save({**checkpoint, "state_dict": model.state_dict()}, predictor_path)
    inference = RewardGuidedResponsePredictor(SimpleNamespace(
        reward_guide_path=guide_path, predictor_path=predictor_path), device="cpu")
    # Exercise the checkpoint-backed inference path with one reverse step to
    # keep the integration test small while still running both neural models
    # and reward guidance on the physical RFV trajectory.
    inference.sampler.T = 1
    history = np.column_stack((np.zeros(20), np.arange(20) * 1.0,
                               np.full(20, 10.), np.zeros(20))).astype(np.float32)
    elcv = trajectory(0., 20.)
    current = trajectory(0., 40., actions=False)
    target = trajectory(3.5, 50., actions=False)
    rng_state = torch.random.get_rng_state()
    predicted = inference.predict(history, elcv, current, target, seed=19)
    repeated = inference.predict(history, elcv, current, target, seed=19)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert predicted.states.shape == (100, 4)
    assert predicted.actions.shape == (100, 2)
    assert np.isfinite(predicted.states).all() and np.isfinite(predicted.actions).all()
    assert np.array_equal(predicted.states, repeated.states)


def test_procedural_warmup_produces_real_history_then_resets_episode_metrics():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv

    env = SRAIRLProceduralEnv("curve", lane_num=3, traffic_density=0., seed=17, horizon=100)
    try:
        env.reset(seed=17)
        initial = env._roles_live()
        history = RoleHistoryBuffer(20)
        history.reset(initial)
        for _ in range(20):
            _, info = env.warmup_step()
            assert not info["done"]
            history.append(env._roles_live())
        actual = history.current()
        assert np.ptp(actual["elcv"][:, 1]) > 10.0
        assert actual["elcv"][-1, 1] > actual["elcv"][0, 1]
        phase = env.begin_evaluation_phase(warmup_steps=20)
        assert phase["warmup_steps"] == 20
        assert env.steps == 0 and env.time == 0.0
        assert env.distance_travelled_m == 0.0
        assert not env._metadrive_collision and not env._metadrive_out_of_road
        expected_start = actual["elcv"][-1, 1]
        assert phase["evaluation_start_s"] == pytest.approx(expected_start)
    finally:
        env.close()


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


def test_rlv_forecast_starts_at_next_timestep():
    from rl.srairl.pipeline.role_forecasting import forecast_rlv_roles

    class Vehicle:
        def __init__(self, x):
            self.position = np.asarray([x, 0.0, 0.0])
            self.velocity = np.asarray([10.0, 0.0, 0.0])
            self.lane = StraightLane()

    class World:
        def __init__(self):
            self.roles = {"rlv_current": Vehicle(20.0), "rlv_target": Vehicle(40.0)}

        def role_vehicle(self, role):
            return self.roles[role]

    class LaneEnv:
        scene_source = "procedural"
        scenario_type = "straight"
        _source_lane = StraightLane()
        world = World()

    history={role:np.tile([0.,x,10.,0.],(20,1))
             for role,x in (('rlv_current',20.),('rlv_target',40.))}
    current, target = forecast_rlv_roles(LaneEnv(), history, horizon=100, dt=0.1)
    assert current.states[0, 1] == pytest.approx(21.0)
    assert current.states[-1, 1] == pytest.approx(120.0)
    assert target.states[0, 1] == pytest.approx(41.0)


def test_joint_candidate_predictor_seed_is_episode_step_and_candidate_id():
    class Predictor:
        def __init__(self):
            self.seeds = []

        def predict(self, rfv_history, elcv_future, rlv_current_future,
                    rlv_target_future, *, seed):
            self.seeds.append(seed)
            return trajectory(0., -12.)

    predictor = Predictor()
    build_joint_candidates([trajectory(), trajectory(1.)], predictor,
                           np.zeros((20, 4)), trajectory(actions=False),
                           trajectory(1., actions=False), seed=2017)
    assert predictor.seeds == [2017, 2018]


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
    with pytest.raises(ValueError, match="source dataset revisions"):
        SRAIRLModelBundle(*paths,
            planner_metadata={"horizon": 100, "source_dataset_revision": "a"},
            predictor_metadata={"horizon": 100, "source_dataset_revision": "b"},
            normalization={"planner_goal": "planner-goal-v1",
                           "reward_predictor": "reward-predictor-v1"})


def test_selector_uses_srairl_rewards_and_rejects_uncalibrated_actions():
    selector = CandidateSelector(LinearReward(48), LinearReward(48),
                                 validated_ax=(-1.5, 1.5), validated_ay=(-1.5, 1.5))
    first = joint(0, 1.0)
    second = joint(1, 2.0)
    rejected = joint(2, 4.0, np.tile([3.0, 0.0], (100, 1)))
    history = {role: np.column_stack((np.full(20, y), np.arange(20),
                                      np.full(20, 10.), np.zeros(20)))
               for role, y in (("elcv", 0.), ("rfv", 3.5),
                               ("rlv_current", 0.), ("rlv_target", 3.5))}
    selected, diagnostics = selector.select([first, second, rejected], history=history)
    assert selected.candidate_id == 1
    assert rejected.feasibility["feasible"] is False
    assert any("outside_calibration" in reason for reason in rejected.feasibility["reasons"])
    assert len(diagnostics) == 3


def test_selector_joins_last_three_observed_frames_to_predicted_frames():
    actual = np.arange(5 * 16, dtype=np.float32).reshape(5, 16)
    predicted = (1000 + np.arange(6 * 16, dtype=np.float32)).reshape(6, 16)
    stacked = CandidateSelector._history_stack(predicted, 4, actual)
    assert stacked.shape == (6, 64)
    assert np.array_equal(stacked[0], np.concatenate([actual[-3], actual[-2], actual[-1], predicted[0]]))
    assert np.array_equal(stacked[1], np.concatenate([actual[-2], actual[-1], predicted[0], predicted[1]]))


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
        def generate_candidates(self, history, planning_context, num_candidates, seed):
            assert all(value.shape == (20, 4) for value in history.values())
            assert planning_context.target_lateral_position == 0.0
            return [trajectory(0.0, 0.0)]

    class Predictor:
        def predict(self, rfv_history, elcv_future, rlv_current_future, rlv_target_future, *, seed):
            assert seed in {9, 1009}
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
        planning_context_provider=lambda env, history: PlanningContext(0.0),
    )
    result = pipeline.evaluate_episode(env, seed=9)
    assert result["metrics"].planning_steps == 2
    assert result["metrics"].selected_candidate_ids == [0, 0]
    assert result["metrics"].lane_change_success
    assert result["warmup_steps"] == 0
    assert result["evaluation_start_states"]["elcv"] == [0., 0., 10., 0.]
    assert result["last_info"]["done"]
