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
    assert model.discriminator_logits(states, dones, log_pi, next_states).shape == (3, 1)
    assert torch.allclose(model.calculate_policy_reward(states,dones,log_pi,next_states),
                          model.discriminator_logits(states,dones,log_pi,next_states))
    model.discriminator_logits(states,dones,log_pi,next_states).sum().backward()
    assert potential._l_long_raw.grad is not None
    assert model.private_g.encoder[0].conv.in_channels==8


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


def _native_component_archive(tmp_path, monkeypatch):
    import json
    import studio.config as config
    monkeypatch.setattr(config,'ROOT',tmp_path)
    root=tmp_path/'datasets'/'srairl'/'metadrive';root.mkdir(parents=True)
    n=121;episodes=3
    states=np.zeros((n*episodes,4,4),dtype=np.float64)
    for episode in range(episodes):
        start=episode*n
        states[start:start+n,0,0]=np.arange(n)*2.
        states[start:start+n,:,2]=20.
    np.savez_compressed(root/'native.npz',role_order=np.asarray(['elcv','rfv','rlv_current','rlv_target']),
        role_states=states,trajectory_times_s=np.tile(np.arange(n)*.1,episodes),
        trajectory_state_offsets=np.arange(episodes+1)*n,
        trajectory_desired_acceleration=np.zeros((episodes*100,2,2),dtype=np.float32),
        trajectory_metadrive_actions=np.zeros((episodes*100,2,2),dtype=np.float32),
        trajectory_actual_acceleration=np.zeros((episodes*100,2,2),dtype=np.float32),
        trajectory_reference_states_frenet=states.copy(),
        trajectory_lane_indices=np.full((n*episodes,4),'main/road/0',dtype='U32'),
        trajectory_action_offsets=np.arange(episodes+1)*100,
        trajectory_metadata=np.asarray(json.dumps([dict(seed=100+i,scenario='straight',warmup_steps=20,
            split=split,road_geometry=__import__('env.tasks.srairl_interaction',fromlist=['ROAD_GEOMETRY_CONFIG']).ROAD_GEOMETRY_CONFIG,teacher_version='curvature-feedback-v3',reference_frame_version='scenario-left-normal-v2') for i,split in enumerate(('train','validation','test'))])))
    return 'datasets/srairl/metadrive/native.npz'


def test_reward_guide_request_requires_a_prior_srairl_model(tmp_path,monkeypatch):
    from studio.config import clean_reward_guide_request
    with pytest.raises(ValueError, match="选择已完成"):
        clean_reward_guide_request({}, "train")
    dataset_id=_native_component_archive(tmp_path,monkeypatch)
    config = clean_reward_guide_request({"srairl_model_id": "models/example",
        "upstream_epochs": 2, "upstream_batch_size": 4, "upstream_seed": 5,
        "upstream_device": "cpu",'srairl_expert_dataset_id':dataset_id}, "train")
    assert config["task"] == "reward_guide"
    assert config["epochs"] == 2
    assert config['dataset_id']==[dataset_id]


def test_upstream_planner_and_predictor_requests_validate_stage_inputs(tmp_path,monkeypatch):
    from studio.config import clean_planner_request, clean_predictor_request
    dataset_id=_native_component_archive(tmp_path,monkeypatch)
    planner = clean_planner_request({"planner_goal_epochs": 2, "planner_dppo_steps": 3,
        "upstream_batch_size": 4, "upstream_seed": 5, "upstream_device": "cpu",
        'srairl_expert_dataset_id':dataset_id}, "train")
    assert (planner["goal_epochs"], planner["dppo_steps"]) == (2, 3)
    with pytest.raises(ValueError, match="闭环评估接口"):
        clean_planner_request({}, "eval")
    predictor = clean_predictor_request({"reward_guide_model_id": "models/guide",
        "upstream_epochs": 2, "upstream_batch_size": 4, "upstream_seed": 5,
        "upstream_device": "cpu",'srairl_expert_dataset_id':dataset_id}, "train")
    assert predictor["task"] == "predictor"
    assert predictor['dataset_id']==[dataset_id]


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
    assert (c.frame_state_dim, c.history_len, c.gamma) == (16, 11, .99)
    assert (c.potential_longitudinal_scale, c.potential_lateral_scale, c.direction_lambda, c.alpha_init) == (15., 3.5, .3, .1)
    assert (c.pos_lat_scale, c.pos_long_scale, c.vel_scale, c.vel_offset) == (25., 500., 80., -40.)
    assert reward_config.FRAME_STATE_DIM == 16


def test_temporal_reward_and_counterfactual_preference_use_full_history():
    import torch
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.discriminator import TemporalPrivateReward, extract_motion_features
    from rl.srairl.trainer import counterfactual_preference_mask
    cfg=SRAIRLConfig.from_mapping()
    model=TemporalPrivateReward()
    features=extract_motion_features(torch.zeros(2,cfg.history_len*16),"lcv",cfg.history_len)
    assert features.shape==(2,8,cfg.history_len)
    assert model(features).shape==(2,1)
    with pytest.raises(ValueError,match="TCN expects"):
        model(torch.zeros(2,4,cfg.history_len))
    states=torch.zeros(1,cfg.history_len,16)
    states[:,:,2]=(10.-cfg.vel_offset)/cfg.vel_scale
    states[:,-1,5]=8./cfg.pos_long_scale
    states[:,-1,6]=-2./cfg.vel_scale
    counterfactual,eligible=counterfactual_preference_mask(states.reshape(1,-1),cfg)
    assert eligible.tolist()==[True]
    assert torch.isclose(counterfactual.reshape(1,cfg.history_len,16)[:,-1,5],
                         torch.tensor([3./cfg.pos_long_scale])).all()


def test_studio_rejects_legacy_trajectory_inputs_for_metadrive_only_protocol():
    from studio.config import clean_request, clean_srairl_request
    with pytest.raises(ValueError, match="无效 data_source"):
        clean_request({"data_source":"ngsim"},"train")
    with pytest.raises(ValueError, match="只使用 MetaDrive"):
        clean_srairl_request({"srairl_scene_source":"ngsim"},"eval")
    with pytest.raises(ValueError, match="不接受轨迹回放"):
        clean_srairl_request({"srairl_scenarios":[{"dataset_id":"legacy","roles":{}}]},"eval")
    with pytest.raises(ValueError, match="MetaDrive 专家示范数据集"):
        clean_srairl_request({"srairl_operation":"interaction"},"train")


def test_expert_collection_accepts_batch_of_three_metadrive_maps():
    from studio.config import clean_srairl_request
    request=clean_srairl_request({"srairl_operation":"expert_collect",
        "srairl_collect_scenarios":["straight","curve","merge"],
        "srairl_demo_episodes":3,"srairl_lane_num":3},"train")
    assert request["scenarios"]==["straight","curve","merge"]
    assert set(request["interaction_scenarios"])==set(request["scenarios"])
    assert set(request['interaction_scenarios']['curve'][key] for key in
        ('elcv_speed','rfv_speed','rlv_current_speed','rlv_target_speed'))=={15.}
    assert request['interaction_scenarios']['merge']['elcv_speed']==20.
    assert request['interaction_scenarios']['merge']['rfv_speed']==22.
    assert request['interaction_scenarios']['merge']['rlv_current_speed']==20.
    assert request['interaction_scenarios']['merge']['rlv_target_speed']==22.
    with pytest.raises(ValueError,match="至少需要三条"):
        clean_srairl_request({"srairl_operation":"expert_collect",
            "srairl_collect_scenarios":["merge"],"srairl_demo_episodes":3,
            "srairl_lane_num":2},"train")


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
        optimizer_ids={id(parameter) for group in trainer.discriminator_optimizer.param_groups
                       for parameter in group['params']}
        assert id(trainer.shared_potential._l_long_raw) in optimizer_ids
        assert isinstance(trainer.bce,torch.nn.BCEWithLogitsLoss)
        history = trainer.train()
        assert len(history) == 1
        assert "discriminator_loss" in history[0]
        assert "preference_rank_loss" in history[0]
        assert "preference_pairs_used" in history[0]
        assert "potential_L_long_m" in history[0]
        assert trainer.episode == 1
        assert (Path(temp) / "training_state.pt").is_file()
        resumed = SRAIRLTrainer([env], expert, config=cfg, training_config=train_cfg)
        resumed.load_training_state(temp)
        assert resumed.episode == trainer.episode
        assert resumed.global_steps == trainer.global_steps
        assert resumed.next_decay_step == trainer.next_decay_step
        for actual, expected in zip(resumed.policy_elcv.policy.parameters(), trainer.policy_elcv.policy.parameters()):
            assert torch.equal(actual, expected)
