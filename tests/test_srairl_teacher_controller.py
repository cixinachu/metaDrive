"""Deterministic behavioral tests for gap-aware teacher decisions and commands."""
from types import SimpleNamespace
import numpy as np
import pytest
from datasets.srairl.teacher_controller import MetaDriveExpertController,TeacherConfig,quintic_progress,braking_margin


class Lane:
    length=1000.
    def __init__(self,y):self.y=y
    def local_coordinates(self,p):return float(p[0]),self.y-float(p[1])
    def position(self,s,lateral=0.):return np.array([s,self.y-lateral])
    def heading_theta_at(self,s):return 0.


def vehicle(x,y,speed=10.,lateral_speed=0.):
    return SimpleNamespace(position=np.array([x,y]),velocity=np.array([speed,lateral_speed]),
        heading_theta=0.,LENGTH=4.5,WIDTH=1.8)


@pytest.fixture
def environment():
    roles={'elcv':vehicle(30,0),'rfv':vehicle(15,3.5),'rlv_current':vehicle(65,0),'rlv_target':vehicle(55,3.5)}
    env=SimpleNamespace(time=0.,steps=0,dt=.1,_is_merge=False,scenario_type='straight',_last_info={},
        _source_lane_path=[Lane(0)],_target_lane_path=[Lane(3.5)],
        interaction_scenario=SimpleNamespace(elcv_speed=10.,rfv_speed=10.,source_lane=0,target_lane=1))
    env.world=SimpleNamespace(role_vehicle=lambda role:roles[role])
    env._vehicle_pool=lambda:list(roles.values())
    env._path_state=lambda v,path:np.array([v.position[1]-path[0].y,v.position[0],*v.velocity])
    env._roles_live=lambda:{key:env._path_state(v,env._source_lane_path) for key,v in roles.items()}
    return env,roles


def test_quintic_has_smooth_endpoints():
    assert quintic_progress(0,8)==(0.,0.,0.)
    assert quintic_progress(8,8)==(1.,0.,0.)
    assert quintic_progress(-1,8)==(0.,0.,0.)
    assert quintic_progress(9,8)==(1.,0.,0.)
    assert quintic_progress(4,8)[0]==pytest.approx(.5)


def test_closing_speed_changes_gap_acceptance():
    cfg=TeacherConfig()
    assert braking_margin(12,10,10,3,cfg)>0
    assert braking_margin(12,14,10,3,cfg)<0


def test_unsafe_gap_holds_lane_then_explicitly_refuses(environment):
    env,roles=environment;roles['rfv'].position[0]=26.
    teacher=MetaDriveExpertController(env)
    for i in (0,1,2,120):
        env.steps=i;env.time=i*.1
        action=teacher.action('elcv')
        assert action[1]==pytest.approx(0.)
    assert teacher.phase=='refused' and teacher.refusal_reason=='gap_wait_timeout'
    assert any(event['phase']=='refused' for event in teacher.events)


def test_safe_gap_requires_confirmation_and_smooth_start(environment):
    env,_=environment;teacher=MetaDriveExpertController(env)
    for i in range(4):
        env.steps=i;env.time=i*.1;action=teacher.action('elcv')
        if i<3:assert teacher.phase=='waiting'
    assert teacher.phase=='executing'
    assert action[1]==pytest.approx(0.)
    env.steps=4;env.time=.4
    assert teacher.action('elcv')[1]>0


def test_rfv_responds_to_predicted_cut_in_before_lane_center(environment):
    env,roles=environment;roles['elcv'].position[1]=1.3
    still=MetaDriveExpertController(env).action('rfv')[0]
    roles['elcv'].velocity[1]=1.
    entering=MetaDriveExpertController(env).action('rfv')[0]
    assert entering<still


def test_action_order_repeatability_and_warmup(environment):
    env,_=environment;a=MetaDriveExpertController(env);b=MetaDriveExpertController(env)
    forward={role:a.action(role) for role in ('elcv','rfv')}
    reverse={role:b.action(role) for role in ('rfv','elcv')}
    for role in forward:np.testing.assert_array_equal(forward[role],reverse[role])
    a.set_warmup(True)
    for i in range(20):
        env.steps=i;env.time=i*.1;a.action('elcv')
    assert a.phase=='warmup'
    env.steps=0;env.time=0.;a.set_warmup(False)
    assert a.phase=='waiting'


def test_acceleration_and_command_jerk_limits(environment):
    env,roles=environment;teacher=MetaDriveExpertController(env)
    previous=np.zeros(2)
    for i in range(60):
        env.steps=i;env.time=i*.1
        command=teacher.action('elcv')
        assert np.max(np.abs(command))<=1.5
        assert np.max(np.abs(command-previous))<=.200001
        np.testing.assert_array_equal(command,teacher.action('elcv'))
        previous=command


def test_invalid_physics_limits_rejected():
    with pytest.raises(ValueError):TeacherConfig(max_acceleration_mps2=3.)
    with pytest.raises(ValueError):TeacherConfig(lane_change_duration_s=.1)


def test_merge_request_can_be_disabled_and_is_not_triggered_for_open_gap(environment):
    env,roles=environment
    env._is_merge=True;env._merge_mainline_path=env._target_lane_path
    teacher=MetaDriveExpertController(env,config=TeacherConfig(merge_gap_request_enabled=False))
    teacher._update()
    assert not teacher.merge_gap_requested
    teacher=MetaDriveExpertController(env)
    teacher._update()
    assert not teacher.merge_gap_requested
    roles['rfv'].velocity[0]=15.
    teacher=MetaDriveExpertController(env)
    teacher._update()
    assert teacher.merge_gap_requested
    assert teacher.phase=='waiting'  # A request never bypasses the gap check.


def test_collection_restores_caller_state_after_callback_error(environment):
    from datasets.srairl.metadrive_expert import collect_metadrive_expert
    env,_=environment;original=env.interaction_scenario
    restored=[];env.world.set_interaction_scenario=lambda value:restored.append(value)
    env._expert_collection=False
    def fail(_):
        env.interaction_scenario=object()
        raise RuntimeError('callback failed')
    with pytest.raises(RuntimeError,match='callback failed'):
        collect_metadrive_expert(env,1,0,fail)
    assert env.interaction_scenario is original and restored[-1] is original
    assert not env._expert_collection


def test_teacher_versions_remain_separate_training_sources(tmp_path):
    import json
    from datasets.srairl.native_windows import native_dataset_info
    from datasets.srairl.teacher_controller import TEACHER_VERSION
    from env.tasks.srairl_interaction import ROAD_GEOMETRY_CONFIG,REFERENCE_FRAME_VERSION
    paths=[]
    for version,base in [('curvature-feedback-v3',0),(TEACHER_VERSION,100)]:
        path=tmp_path/(version+'.npz');states=np.zeros((363,4,4));actions=np.zeros((303,2,2))
        metadata=[dict(seed=base+i,scenario='straight',warmup_steps=20,teacher_version=version,
            road_geometry=ROAD_GEOMETRY_CONFIG,reference_frame_version=REFERENCE_FRAME_VERSION) for i in range(3)]
        np.savez_compressed(path,role_order=np.array(['elcv','rfv','rlv_current','rlv_target']),
            role_states=states,trajectory_times_s=np.tile(np.arange(121)*.1,3),
            trajectory_state_offsets=np.array([0,121,242,363]),trajectory_action_offsets=np.array([0,101,202,303]),
            trajectory_desired_acceleration=actions,trajectory_metadrive_actions=actions,
            trajectory_actual_acceleration=actions,trajectory_reference_states_frenet=states,
            trajectory_lane_indices=np.full((363,4),'lane'),trajectory_metadata=np.asarray(json.dumps(metadata)))
        assert native_dataset_info(path,require_current=True)['teacher_versions']==[version]
        paths.append(path)
    with pytest.raises(ValueError,match='teacher versions cannot be mixed'):
        native_dataset_info(paths,require_current=True)


def test_physical_deceleration_never_maps_to_positive_throttle():
    from env.adapters.srairl_action import SRAIRLActionAdapter
    adapter=SRAIRLActionAdapter()
    adapter.calibration=dict(speeds=np.array([5.,10.]),controls=np.array([-.1,-.002,0.,.002,.1]),
        longitudinal_accel=np.array([[-2.7,-.054,-.364,.006,.29]]*2),
        lateral_accel=np.array([[-1.,-.02,0.,.02,1.]]*2))
    v=vehicle(0,0,8.)
    for acceleration in (-1.,-.285,-.01):
        assert adapter.to_metadrive(v,[acceleration,0.],.1)[1]<0
    assert adapter.to_metadrive(v,[0.,0.],.1)[1]>0


def test_stopped_obstacle_braking_and_triggered_traffic_perception(environment):
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    env,roles=environment
    obstacle=vehicle(100,0,0)
    roles['rlv_current']=obstacle
    command=MetaDriveExpertController(env).action('elcv')
    assert command[0]<0  # Anticipatory braking, even with a large static gap.
    a,b,c=object(),object(),object()
    world=SimpleNamespace(agents={'a':a},interaction_vehicles={'b':b},
        engine=SimpleNamespace(traffic_manager=SimpleNamespace(traffic_vehicles=[c],spawned_objects={'b':b,'c':c,'parked':obstacle})))
    perceived=SRAIRLProceduralEnv._vehicle_pool(SimpleNamespace(world=world))
    assert len(perceived)==4 and any(v is obstacle for v in perceived)


def test_requested_rfv_response_is_distinct_from_neutral_cruise():
    from datasets.srairl.metadrive_expert import validate_expert_episode
    t=np.arange(141)*.1-2
    states=np.zeros((141,4,4));states[:,:,2]=10
    states[:,:,0]=10*t[:,None]+np.array([30,15,65,55])[None,:]
    reference=states.copy();reference[:,:,1]=states[:,:,0]
    actions=np.zeros((120,2,2))
    raw=dict(role_order=['elcv','rfv','rlv_current','rlv_target'],role_states=states,
        reference_states_frenet=reference,times_s=t,lane_indices=np.full((141,4),'lane'),
        desired_acceleration=actions.copy(),metadrive_actions=actions.copy(),actual_acceleration=actions.copy(),
        tracking_errors=actions.copy(),warmup_steps=20,lane_change_completion_index=40,lane_change_success=True)
    assert 'rfv_actuation_response_missing' not in validate_expert_episode(raw)['errors']
    raw['desired_acceleration'][:,1,0]=-.3
    assert 'rfv_actuation_response_missing' in validate_expert_episode(raw)['errors']
    raw['reference_states_frenet'][40,3,1]=0.
    assert 'target_slot_not_preserved' in validate_expert_episode(raw)['errors']
    raw['desired_acceleration']=np.zeros(120)
    assert 'action_alignment' in validate_expert_episode(raw)['errors']


def test_warmup_does_not_extend_interaction_beyond_world_horizon():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    env=SRAIRLProceduralEnv('straight',3,0.,0,horizon=30)
    try:
        env.reset(seed=0)
        for _ in range(20):env.step({'elcv':np.zeros(2),'rfv':np.zeros(2)})
        env.begin_evaluation_phase(20)
        assert env.max_steps==30
        for _ in range(30):_,info=env.step({'elcv':np.zeros(2),'rfv':np.zeros(2)})
        assert info['done'] and env.steps==30
        assert env.world.role_vehicle('elcv') is not None
    finally:env.close()


def test_native_traffic_activates_for_agents_spawned_inside_merge_block():
    from env.tasks.srairl_metadrive_world import activate_occupied_traffic_blocks
    car=object();future_car=object()
    current=SimpleNamespace(trigger_road='r_entry',vehicles=['current'])
    future=SimpleNamespace(trigger_road='next_entry',vehicles=['future'])
    def block(entry,start,end):
        return SimpleNamespace(pre_block_socket=SimpleNamespace(positive_road=entry),
            block_network=SimpleNamespace(graph={start:{end:[]}}))
    manager=SimpleNamespace(engine=SimpleNamespace(agent_manager=SimpleNamespace(
        active_agents={'elcv':SimpleNamespace(lane_index=('ramp_in','ramp_out',0))})),
        current_map=SimpleNamespace(blocks=[None,block('r_entry','ramp_in','ramp_out'),block('next_entry','next_in','next_out')]),
        block_triggered_vehicles=[future,current],_traffic_vehicles=[],
        get_objects=lambda ids:{key:{'current':car,'future':future_car}[key] for key in ids})
    activate_occupied_traffic_blocks(manager)
    assert manager._traffic_vehicles==[car]
    assert manager.block_triggered_vehicles==[future]
    activate_occupied_traffic_blocks(manager)
    assert manager._traffic_vehicles==[car]
