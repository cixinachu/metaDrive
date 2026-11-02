import numpy as np
import pytest
from pathlib import Path
import tempfile

from env.core.types import InteractionScenario
from env.adapters.srairl_observation import SRAIRLObservationAdapter
from env.tasks.srairl_env import SRAIRLEnv
from tasks.registry import TASK_REGISTRY


def scenario(duration=.4):
    times = np.arange(-.2, duration+.01, .1)
    def track(ident, x_offset, y_offset):
        states = [[t, x_offset + 10*t, y_offset, 10., 0.] for t in times]
        return {"id": ident, "states": states, "length": 4.5, "width": 1.8}
    tracks = {v["id"]: v for v in [track("elcv", 0, 3), track("rfv", -10, 3),
                                      track("rlvc", 15, 3), track("rlvt", 15, 6.5)]}
    return InteractionScenario("s1", "elcv", "rfv", tracks, 3, 3.5, 100, "train", "ngsim",
                               "rlvc", "rlvt")


def test_canonical_scenario_requires_explicit_role_tracks():
    with pytest.raises(ValueError, match="missing"):
        InteractionScenario("bad", "a", "b", {}, 3, 3.5, 100, "train", "ngsim")


def test_observation_reproduces_upstream_16d_relative_frame_and_history_padding():
    roles = {"elcv": [3.5, 100, 20, 0], "rfv": [3.5, 90, 19, 0],
             "rlv_current": [3.5, 115, 21, 0], "rlv_target": [7, 120, 20, 0]}
    adapter = SRAIRLObservationAdapter(history_len=4)
    obs = adapter.reset(roles)
    assert obs.shape == (64,)
    assert np.array_equal(obs[:16], obs[-16:])
    frame = adapter.frame(roles)
    assert np.allclose(frame[4:8], adapter.normalize_vehicle(roles["elcv"]) - adapter.normalize_vehicle(roles["rfv"]))


def test_srairl_kinematics_matches_upstream_backward_differences():
    from datasets.srairl.kinematics import acceleration, state
    times=np.arange(-.2,.31,.1)
    track={"id":"known-motion","states":[[t,20.+10.*t+2.*t*t,3.,0.,0.] for t in times]}
    assert state(track,0.)[2] == pytest.approx(9.8)
    assert acceleration(track,0.)[0] == pytest.approx(4.)


def test_legacy_environment_uses_updated_velocity_for_position_and_joint_obs():
    env = SRAIRLEnv(scenario(), history_len=1)
    obs, info = env.reset()
    assert obs["elcv"].shape == (16,)
    next_obs, info = env.step({"elcv": [2., 0.], "rfv": [0., 0.]})
    state = env.vehicles["elcv"]
    assert state["v_long"] == pytest.approx(10.2)
    assert state["lon"] == pytest.approx(.1 * 10.2)
    assert next_obs["elcv"].shape == (16,)
    assert info["dynamics_mode"] == "legacy_kinematic"


def test_physics_mode_requires_a_real_two_agent_metadrive_world():
    with pytest.raises(ValueError, match="two-agent MetaDrive world"):
        SRAIRLEnv(scenario(), dynamics_mode="metadrive_physics")


def test_metadrive_physics_controls_both_interaction_actors():
    from studio.worker import create_srairl_metadrive_world
    scene=scenario(.8)
    world=create_srairl_metadrive_world(scene,"train","topdown",True,"metadrive_physics")
    env=SRAIRLEnv(scene,dynamics_mode="metadrive_physics",metadrive_world=world)
    try:
        for speed in (5.,10.,20.,30.):
            env.reset()
            assert world.is_multi_agent and set(world.agents)=={"agent0","agent1"}
            for role in ("elcv","rfv"):
                vehicle=world.role_vehicle(role)
                vehicle.set_velocity((speed,0.))
                vehicle.set_heading_theta(0.)
            before={role:env._read_world_state(world.role_vehicle(role))["v_long"] for role in ("elcv","rfv")}
            before_x={role:float(world.role_vehicle(role).position[0]) for role in ("elcv","rfv")}
            for _ in range(5):
                _,info=env.step({"elcv":[1.,0.],"rfv":[-1.,0.]})
            actual_positive=(env.vehicles[scene.elcv_id]["v_long"]-before["elcv"])/.5
            actual_braking=(env.vehicles[scene.rfv_id]["v_long"]-before["rfv"])/.5
            assert actual_braking < 0
            if speed < 30.:
                assert actual_positive > 0
                if speed == 10.:
                    # This recorded-map harness checks direction and rough
                    # scale. Quantitative calibration uses the procedural map.
                    assert actual_positive == pytest.approx(1.,abs=.4)
                    assert actual_braking == pytest.approx(-1.,abs=.5)
            assert world.role_vehicle("elcv").position[0] > before_x["elcv"]
            assert world.role_vehicle("rfv").position[0] > before_x["rfv"]
            assert info["dynamics_mode"]=="metadrive_physics"
    finally:
        env.close()


def test_legacy_controlled_pair_synchronizes_to_shared_metadrive_world():
    from studio.worker import create_srairl_metadrive_world
    scene=scenario()
    world=create_srairl_metadrive_world(scene,"train","topdown")
    env=SRAIRLEnv(scene,metadrive_world=world)
    try:
        env.reset()
        assert len(world.engine.traffic_manager.traffic_vehicles)==2
        _,info=env.step({"elcv":[1.,0.],"rfv":[-1.,0.]})
        assert np.asarray(world.agent.position)[:2] == pytest.approx(
            [env.vehicles[scene.elcv_id]["lon"],env.vehicles[scene.elcv_id]["lat"]])
        assert np.asarray(env._meta_rfv.position)[:2] == pytest.approx(
            [env.vehicles[scene.rfv_id]["lon"],env.vehicles[scene.rfv_id]["lat"]])
        assert info["dynamics_mode"]=="legacy_kinematic"
        frame=world.render(mode="topdown",window=False,screen_size=(320,240),film_size=(640,480),
                           num_stack=3,draw_target_vehicle_trajectory=False,scaling=12)
        assert np.asarray(frame).shape[:2]==(240,320)
    finally:
        env.close()


def test_reactive_recorded_traffic_changes_speed_when_policy_vehicle_closes_gap():
    from env.recorded_traffic import InteractionTrafficManager
    class Engine:
        global_config={"physics_world_step_size":.1}
    class Actor:
        position=np.asarray([15.,3.])
        speed=8.
    track={"width":1.8}
    nominal=np.asarray([0.,3.,20.,0.])
    changed=InteractionTrafficManager.reactive_state_for_track(track,nominal,[Actor()],.1,.7,5.)
    assert changed[2] < nominal[2]
    assert changed[0] < nominal[0]


def test_task_registry_keeps_algorithm_family_separate():
    assert TASK_REGISTRY["safe_rl"].status == "available"
    assert TASK_REGISTRY["srairl"].status == "available"
    assert TASK_REGISTRY["planner"].status == "available"
    assert TASK_REGISTRY["reward_guide"].status == "available"
    assert TASK_REGISTRY["predictor"].status == "available"


def test_social_airl_discriminator_uses_checked_upstream_reward_shape():
    torch = pytest.importorskip("torch")
    from rl.srairl.discriminator import AnisotropicInteractionPotential, SocialAIRLDiscrim
    potential = AnisotropicInteractionPotential(L_long=15., L_lat=3.5, dir_lambda=.3)
    model = SocialAIRLDiscrim(16, .99, potential, alpha_init=.1, role="lcv", history_len=1)
    states = torch.zeros((3, 16), dtype=torch.float32)
    next_states = torch.ones((3, 16), dtype=torch.float32) * .1
    dones = torch.tensor([False, False, True])
    log_pi = torch.zeros(3)
    reward, private, phi = model.get_reward(states)
    assert reward.shape == private.shape == phi.shape == (3, 1)
    assert model(states, dones, log_pi, next_states).shape == (3, 1)
    assert torch.allclose(reward, private - model.alpha * phi)


def test_social_potential_denormalizes_velocity_with_the_supplied_offset():
    torch = pytest.importorskip("torch")
    from rl.srairl.discriminator import AnisotropicInteractionPotential
    p = AnisotropicInteractionPotential(15., 3.5, .3, pos_lat_scale=25.,pos_long_scale=500.,vel_scale=80.,vel_offset=-40.)
    # A normalized velocity of 0.5 maps back to 0 m/s: .5*80 + (-40).
    physical = p._physical_state(torch.tensor([[.2,.4,.5,.5]],dtype=torch.float32))
    assert [float(v.item()) for v in physical] == pytest.approx([5.,200.,0.,0.])


def test_srairl_state_dict_checkpoint_loads_as_upstream_reward_guide():
    torch = pytest.importorskip("torch")
    import json
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.discriminator import AnisotropicInteractionPotential, SocialAIRLDiscrim
    from rl.srairl.upstream.adapter import load_reward_discriminator
    cfg = SRAIRLConfig.from_mapping()
    dim = cfg.frame_state_dim * cfg.history_len
    potential = AnisotropicInteractionPotential(cfg.potential_longitudinal_scale,
        cfg.potential_lateral_scale, cfg.direction_lambda, pos_lat_scale=cfg.pos_lat_scale,
        pos_long_scale=cfg.pos_long_scale, vel_scale=cfg.vel_scale, vel_offset=cfg.vel_offset)
    model = SocialAIRLDiscrim(dim, cfg.gamma, potential, cfg.alpha_init, role="fv",
                              history_len=cfg.history_len)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        (path / "config.json").write_text(json.dumps({"algorithm": "SR-AIRL", "srairl": cfg.__dict__}))
        torch.save(model.state_dict(), path / "discriminator_rfv.pt")
        restored = load_reward_discriminator(path, "cpu")
        reward = restored.get_reward(torch.zeros((2, dim)))[0]
        assert reward.shape == (2, 1)
        assert not restored.training


def test_reward_guide_request_requires_a_prior_srairl_model():
    from studio.config import clean_reward_guide_request
    with pytest.raises(ValueError, match="选择已完成"):
        clean_reward_guide_request({}, "train")
    config = clean_reward_guide_request({"srairl_model_id": "models/example",
        "upstream_epochs": 2, "upstream_batch_size": 4, "upstream_seed": 5,
        "upstream_device": "cpu"}, "train")
    assert config["task"] == "reward_guide"
    assert config["epochs"] == 2


def test_upstream_planner_and_predictor_requests_validate_stage_inputs():
    from studio.config import clean_planner_request, clean_predictor_request
    planner = clean_planner_request({"planner_goal_epochs": 2, "planner_dppo_steps": 3,
        "upstream_batch_size": 4, "upstream_seed": 5, "upstream_device": "cpu"}, "train")
    assert (planner["goal_epochs"], planner["dppo_steps"]) == (2, 3)
    with pytest.raises(ValueError, match="闭环评估接口"):
        clean_planner_request({}, "eval")
    predictor = clean_predictor_request({"reward_guide_model_id": "models/guide",
        "upstream_epochs": 2, "upstream_batch_size": 4, "upstream_seed": 5,
        "upstream_device": "cpu"}, "train")
    assert predictor["task"] == "predictor"


def test_expert_builder_keeps_two_actor_actions_separate_and_requires_preroll():
    from datasets.srairl.expert_builder import ExpertDatasetBuilder
    transitions = ExpertDatasetBuilder(history_len=2).build(scenario())
    assert len(transitions) == 8
    assert {t.agent for t in transitions} == {"elcv", "rfv"}
    assert transitions[0].state.shape == (32,)
    assert transitions[0].action.shape == (2,)
    assert sum(t.done for t in transitions) == 2


def test_supplied_reward_config_values_are_the_sr_airl_defaults():
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl import reward_config
    c = SRAIRLConfig.from_mapping()
    assert (c.frame_state_dim, c.history_len, c.gamma) == (16, 4, .99)
    assert (c.potential_longitudinal_scale, c.potential_lateral_scale, c.direction_lambda, c.alpha_init) == (15., 3.5, .3, .1)
    assert (c.pos_lat_scale, c.pos_long_scale, c.vel_scale, c.vel_offset) == (25., 500., 80., -40.)
    assert reward_config.FRAME_STATE_DIM == 16


def test_studio_srairl_request_requires_explicit_distinct_ngsim_roles(monkeypatch):
    from studio.config import clean_srairl_request
    import datasets.importer
    times=np.arange(-.2,.41,.1)
    offsets=[(0,0),(-10,3.5),(15,0),(15,3.5)]
    tracks=[{"id":str(i),"states":[[t,x+10*t,y,10.,0.] for t in times],"length":4.5,"width":1.8}
            for i,(x,y) in enumerate(offsets)]
    for row in tracks[0]["states"]:
        if row[0]>.0:row[2]=3.5
    clip={"tracks":tracks,"lane_num":3,"lane_width":3.5,"road_length":100}
    monkeypatch.setattr(datasets.importer,"load_clip",lambda _id:(clip,{"source":"ngsim","split":"train","id":"a"*24}))
    request=clean_srairl_request({"srairl_scenarios":[{"dataset_id":"a"*24,"roles":{
        "elcv_id":"0","rfv_id":"1","rlv_current_id":"2","rlv_target_id":"3"}}],
        "srairl_episodes":2,"srairl_ppo_epochs":2,"srairl_seed":0,"srairl_device":"cpu"},"train")
    assert request["dynamics_mode"]=="legacy_kinematic"
    assert request["srairl_reactive_background"] is False
    with pytest.raises(ValueError,match="不同轨迹"):
        clean_srairl_request({"srairl_scenarios":[{"dataset_id":"a"*24,"roles":{
            "elcv_id":"0","rfv_id":"0","rlv_current_id":"2","rlv_target_id":"3"}}]},"train")
    monkeypatch.setattr(datasets.importer,"load_clip",lambda _id:({"tracks":tracks},{"source":"ngsim","split":"eval","id":"a"*24}))
    with pytest.raises(ValueError,match="train 数据片段"):
        clean_srairl_request({"srairl_scenarios":[{"dataset_id":"a"*24,"roles":{
            "elcv_id":"0","rfv_id":"1","rlv_current_id":"2","rlv_target_id":"3"}}]},"train")


def test_srairl_trainer_runs_one_alternating_update_end_to_end():
    import torch
    from datasets.srairl.expert_builder import ExpertDatasetBuilder
    from rl.srairl.trainer import SRAIRLTrainer, SRAIRLTrainingConfig
    from rl.srairl.config import SRAIRLConfig
    cfg = SRAIRLConfig.from_mapping()
    scene = scenario()
    expert = ExpertDatasetBuilder(config=cfg).build(scene)
    env = SRAIRLEnv(scene, config=cfg)
    train_cfg = SRAIRLTrainingConfig(episodes=1, ppo_epochs=2, discriminator_every_episodes=1,
        discriminator_batch_size=2, discriminator_updates=1, lambda_equilibrium=0., device="cpu")
    with tempfile.TemporaryDirectory() as temp:
        trainer = SRAIRLTrainer([env], expert, config=cfg, training_config=train_cfg, output_dir=temp)
        history = trainer.train()
        assert len(history) == 1
        assert "discriminator_loss" in history[0]
        assert trainer.episode == 1
        assert (Path(temp) / "training_state.pt").is_file()
        resumed = SRAIRLTrainer([env], expert, config=cfg, training_config=train_cfg)
        resumed.load_training_state(temp)
        assert resumed.episode == trainer.episode
        assert resumed.global_steps == trainer.global_steps
        assert resumed.next_decay_step == trainer.next_decay_step
        for actual, expected in zip(resumed.policy_elcv.policy.parameters(), trainer.policy_elcv.policy.parameters()):
            assert torch.equal(actual, expected)
