import numpy as np
import pytest

from env.core.types import InteractionScenario
from env.tasks.srairl_interaction import SRAIRLInteractionScenario
from datasets.srairl.interaction_validator import validate_interaction_scenario,extract_initial_state_distribution


def test_lane_change_template_requires_adjacent_target_lane():
    assert SRAIRLInteractionScenario().validate(2).target_lane == 1
    with pytest.raises(ValueError,match='adjacent'):
        SRAIRLInteractionScenario(target_lane=2).validate(3)


def test_ngsim_roles_must_form_a_real_gap_acceptance_event():
    times=np.arange(0.,1.1,.1)
    def track(ident,x,y):
        rows=np.asarray([[t,x+10*t,y,10.,0.] for t in times])
        return {'id':ident,'length':4.5,'width':1.8,'states':rows}
    elcv=np.asarray([[t,10.+10*t,0. if t<.5 else 3.7,10.,0.] for t in times])
    tracks={'e':{'id':'e','length':4.5,'width':1.8,'states':elcv},
        'f':track('f',0.,3.7),'c':track('c',40.,0.),'t':track('t',35.,3.7)}
    scenario=InteractionScenario('synthetic','e','f',tracks,2,3.7,200.,'test','NGSIM',
        'c','t',start_time=0.,end_time=1.)
    result=validate_interaction_scenario(scenario)
    assert result['valid'] and result['source_lane']==0 and result['target_lane']==1
    assert result['initial_state']['elcv_rfv_gap_m']==pytest.approx(10.)
    samples=extract_initial_state_distribution(scenario,samples=4,window_s=.3)
    assert len(samples)==4
    assert all(sample['time_s']<result['lane_change_time_s'] for sample in samples)
    assert all(sample['elcv_rfv_gap_m']>0 and sample['rlv_target_gap_m']>0 for sample in samples)
    assert SRAIRLInteractionScenario.from_ngsim_initial_state(samples[0]).validate(2).target_lane==1
    tracks['f']=track('f',30.,3.7)
    bad=InteractionScenario('synthetic-bad','e','f',tracks,2,3.7,200.,'test','NGSIM','c','t',end_time=1.)
    with pytest.raises(ValueError,match='behind ELCV'):
        validate_interaction_scenario(bad)


def test_procedural_scene_builds_four_roles_and_zero_action_does_not_merge():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    env=SRAIRLProceduralEnv('straight',2,0.,seed=91,horizon=40)
    try:
        env.reset()
        roles=env._roles_live()
        assert set(roles)=={'elcv','rfv','rlv_current','rlv_target'}
        assert env.world.role_vehicle('elcv').lane_index[2]==0
        assert env.world.role_vehicle('rfv').lane_index[2]==1
        assert env._initial_gap==pytest.approx(40.,abs=.1)
        for _ in range(30):
            _,info=env.step({'elcv':np.zeros(2),'rfv':np.zeros(2)})
            if info['done']:break
        assert not info['lane_change_completed']
        assert not info['merge_success']
    finally:
        env.close()


def test_rule_gap_acceptance_teacher_can_complete_lane_change():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    from datasets.srairl.metadrive_expert import MetaDriveExpertController
    from dataclasses import replace
    env=SRAIRLProceduralEnv('straight',2,0.,seed=93,horizon=120,
        interaction_scenario=replace(SRAIRLInteractionScenario(),interaction_end_x=250.))
    try:
        if env.action_adapter.calibration is None:
            pytest.skip('the measured MetaDrive action calibration artifact is not installed')
        observations,_=env.reset();teacher=MetaDriveExpertController(env);info={}
        while not info.get('done',False):
            actions={role:teacher.action(role) for role in ('elcv','rfv')}
            observations,info=env.step(actions)
        assert info['lane_change_completed']
        assert info['merge_success']
        assert not info['metadrive_primary_collision']
        assert not info['metadrive_out_of_road']
        assert info['time_to_lane_change']>0.
    finally:
        env.close()


def test_scripted_rfv_responses_change_gap_dynamics():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    env=SRAIRLProceduralEnv('straight',2,0.,seed=92,horizon=20)
    speeds={}
    try:
        for mode in ('fixed_speed','scripted_yield','scripted_gap_closing'):
            env.rfv_response=mode;env.reset();env._lane_change_started=True
            for _ in range(4):env.step({'elcv':np.zeros(2),'rfv':np.zeros(2)})
            speeds[mode]=float(np.linalg.norm(env.world.role_vehicle('rfv').velocity))
        assert speeds['scripted_yield'] < speeds['fixed_speed'] < speeds['scripted_gap_closing']
    finally:
        env.close()


def test_scripted_yield_and_gap_closing_change_lane_change_outcome():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    from datasets.srairl.metadrive_expert import MetaDriveExpertController

    outcomes={}
    for response in ('fixed_speed','scripted_yield','scripted_gap_closing'):
        env=SRAIRLProceduralEnv('straight',2,0.,seed=110,horizon=200,rfv_response=response)
        try:
            observations,_=env.reset()
            teacher=MetaDriveExpertController(env)
            info={}
            while not info.get('done',False):
                actions={role:teacher.action(role) for role in ('elcv','rfv')}
                observations,info=env.step(actions)
            outcomes[response]=info
        finally:
            env.close()

    fixed=outcomes['fixed_speed']
    yielding=outcomes['scripted_yield']
    closing=outcomes['scripted_gap_closing']
    assert yielding['merge_success'] and not yielding['metadrive_primary_collision']
    assert not closing['merge_success'] and closing['metadrive_primary_collision']
    assert yielding['target_gap_change_m'] > fixed['target_gap_change_m']
    assert yielding['rfv_min_accel_mps2'] == pytest.approx(-1.,abs=.02)
    assert closing['target_gap_change_m'] < fixed['target_gap_change_m']


def test_metadrive_ramp_and_mainline_share_conflict_route_and_no_false_merge_success():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    env=SRAIRLProceduralEnv('merge',3,0.,seed=4,horizon=100)
    try:
        env.reset(seed=4);world=env.world;ego=world.role_vehicle('elcv');rfv=world.role_vehicle('rfv')
        assert ego.lane_index[0].startswith('1r1_')
        assert rfv.lane_index[:2]==('>>>','1r0_0_')
        assert ego.navigation.final_lane is rfv.navigation.final_lane
        assert '1r0_0_' in world.current_map.road_network.graph['1r1_3_']
        assert '1r0_0_' in world.current_map.road_network.graph['>>>']
        for _ in range(100):
            actions={role:env._to_metadrive_action(world.role_vehicle(role),np.zeros(2)) for role in ('elcv','rfv')}
            _,info=env.step(actions)
            if info['done']:break
        assert info['done']
        # If ramp momentum completes a merge without steering, it still
        # must satisfy the generated route and collision/off-road checks.
        if info['merge_success']:
            assert info['merge_route_valid']
            assert not info['metadrive_primary_collision']
            assert not info['metadrive_out_of_road']
        elif info['lane_change_completed']:
            assert not info['merge_shared_lane']
        if env.action_adapter.calibration is not None:
            from datasets.srairl.metadrive_expert import MetaDriveExpertController
            env.rfv_response='fixed_speed';env.reset(seed=4)
            teacher=MetaDriveExpertController(env)
            for _ in range(100):
                _,info=env.step({role:teacher.action(role) for role in ('elcv','rfv')})
                if info['done']:break
            assert info['merge_success']
            assert info['merge_route_valid']
            assert not info['metadrive_primary_collision']
            assert not info['metadrive_out_of_road']
    finally:
        env.close()


def test_ngsim_initial_state_distribution_reconfigures_each_meta_drive_episode():
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    samples=[
        {'time_s':1.,'elcv_rfv_gap_m':10.,'rlv_current_gap_m':35.,'rlv_target_gap_m':25.,
         'elcv_speed_mps':20.,'rfv_speed_mps':19.,'rlv_current_speed_mps':18.,'rlv_target_speed_mps':18.,
         'source_lane':0,'target_lane':1},
        {'time_s':2.,'elcv_rfv_gap_m':15.,'rlv_current_gap_m':40.,'rlv_target_gap_m':30.,
         'elcv_speed_mps':21.,'rfv_speed_mps':18.,'rlv_current_speed_mps':17.,'rlv_target_speed_mps':19.,
         'source_lane':1,'target_lane':0},
    ]
    env=SRAIRLProceduralEnv('straight',2,0.,seed=2,horizon=50,initial_state_samples=samples)
    try:
        selected=set()
        for _ in range(8):
            env.reset();selected.add(env.current_initial_state['time_s'])
            assert env.world.role_vehicle('elcv').lane_index[2]==env.interaction_scenario.source_lane
            assert env.world.role_vehicle('rfv').lane_index[2]==env.interaction_scenario.target_lane
            assert env._initial_gap==pytest.approx(env.interaction_scenario.target_gap_m)
        assert selected=={1.,2.}
    finally:
        env.close()


def test_metadrive_expert_archive_serializes_tracking_errors_and_episode_splits(tmp_path):
    import json
    from datasets.srairl.expert_builder import ExpertTransition,save_transitions
    from datasets.srairl.native_windows import native_dataset_info

    transitions=[];raw_episodes=[]
    for seed in (10,11,12):
        scenario_id=f'metadrive-straight-seed{seed}'
        state=np.zeros(16*11,dtype=np.float32)
        transitions.extend([
            ExpertTransition(state,np.zeros(2,dtype=np.float32),state,True,role,scenario_id,0)
            for role in ('elcv','rfv')])
        raw_episodes.append({
            'role_order':['elcv','rfv','rlv_current','rlv_target'],
            'role_states':np.zeros((121,4,4),dtype=np.float64),
            'reference_states_frenet':np.zeros((121,4,4),dtype=np.float64),
            'lane_indices':[['a/b/0']*4 for _ in range(121)],
            'times_s':np.arange(121,dtype=np.float64)*.1,
            'desired_acceleration':np.zeros((100,2,2),dtype=np.float32),
            'metadrive_actions':np.zeros((100,2,2),dtype=np.float32),
            'actual_acceleration':np.zeros((100,2,2),dtype=np.float32),
            'tracking_errors':np.zeros((100,2,2),dtype=np.float32),
            'warmup_steps':20,'scenario':'straight','seed':seed,
            'validation':{'valid':True,'errors':[]},
        })
    path=tmp_path/'experts.npz'
    save_transitions(path,transitions,{'source':'MetaDrive'},raw_episodes)
    with np.load(path,allow_pickle=False) as archive:
        assert archive['trajectory_tracking_errors'].shape==(300,2,2)
        episode_metadata=json.loads(str(archive['trajectory_metadata'].item()))
        assert len(episode_metadata)==3
        assert all('tracking_errors' not in row for row in episode_metadata)
    info=native_dataset_info(path)
    assert set(info['windows_by_scenario']['straight'])=={'train','validation','test'}
