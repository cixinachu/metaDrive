import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from studio.config import _srairl_interaction_for_scenario
from env.tasks.srairl_interaction import SCENARIO_SPEED_CONFIG, SRAIRLInteractionScenario


def test_scenario_speed_defaults_and_isolated_overrides():
    raw={'srairl_interaction_elcv_speed': 6}
    for scene, speeds in SCENARIO_SPEED_CONFIG.items():
        result=_srairl_interaction_for_scenario(raw,3,scene)
        assert all(result[k]==v for k,v in speeds.items())
    raw.update(srairl_custom_speeds=True,srairl_speed_overrides={'curve':{'elcv_speed':20}})
    assert _srairl_interaction_for_scenario(raw,3,'curve')['elcv_speed']==20
    assert _srairl_interaction_for_scenario(raw,3,'merge')['elcv_speed']==SCENARIO_SPEED_CONFIG['merge']['elcv_speed']


@pytest.mark.parametrize('speed',[float('nan'),float('inf'),-1,0,26])
def test_speed_range(speed):
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(SRAIRLInteractionScenario(),elcv_speed=speed).validate(3)


def test_upstream_numerical_equivalence_and_potential_update(monkeypatch):
    from rl.srairl import reward_config as constants
    from rl.srairl.discriminator import AnisotropicInteractionPotential, SocialAIRLDiscrim, extract_motion_features
    monkeypatch.setitem(sys.modules,'reward_config',constants)
    spec=importlib.util.spec_from_file_location('srairl_test_oracle',Path(__file__).parent/'fixtures/srairl_upstream/airl.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    torch.manual_seed(77)
    states=torch.rand(4,176);next_states=torch.rand(4,176)
    dones=torch.tensor([0.,1.,0.,1.]);logpis=torch.randn(4)
    potential=AnisotropicInteractionPotential(15.,3.5,.3)
    reference_potential=upstream.AnisotropicInteractionPotential(15.,3.5,.3)
    for role in ('lcv','fv'):
        model=SocialAIRLDiscrim(176,.99,potential,.1,role=role)
        reference=upstream.SocialAIRLDiscrim(176,.99,reference_potential,role=role)
        reference.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(extract_motion_features(states,role),upstream.extract_motion_features(states,role))
        for actual,expected in zip(model.get_reward(states),reference.get_reward(states)):
            torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(model.calculate_policy_reward(states,dones,logpis,next_states),reference.calculate_policy_reward(states,dones,logpis,next_states))
    optimizer=torch.optim.Adam(potential.parameters(),lr=.01)
    before=[p.detach().clone() for p in potential.parameters()]
    model.get_reward(states)[0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in potential.parameters())
    optimizer.step()
    assert all(not torch.equal(old,new) for old,new in zip(before,potential.parameters()))


def test_calibration_gate_recomputes_acceptance():
    from studio.worker import action_calibration_passed
    metrics={axis:{split:dict(passed=True,mae_mps2=.01,max_error_mps2=.02,wrong_sign_count=0)
                   for split in ('calibration_grid','held_out')} for axis in ('longitudinal','lateral')}
    report=dict(equivalence_status='passed',metrics=metrics)
    assert action_calibration_passed(report)
    metrics['lateral']['held_out']['mae_mps2']=.5
    assert not action_calibration_passed(report)


def test_browser_scene_switch_uses_api_defaults_and_keeps_custom_values():
    import shutil
    import subprocess
    if not shutil.which('node'):pytest.skip('node is unavailable')
    source=(Path(__file__).resolve().parents[1]/'studio/static/app.js').read_text()
    function=source[source.index('function syncScenarioSpeeds()'):source.index('function interactionSettings(')]
    script="""
const elements={srairl_scenario:{value:'straight'},srairl_custom_speeds:{checked:false},srairlSpeedSummary:{}};
const $=id=>elements[id]??=( {} );
const catalog={srairl_scenario_speeds:DEFAULTS};
const speedOverrides={curve:{elcv_speed:19}};
FUNCTION
for(const [scene,speeds] of Object.entries(catalog.srairl_scenario_speeds)){
 elements.srairl_scenario.value=scene;syncScenarioSpeeds();
 for(const [key,value] of Object.entries(speeds))if($('srairl_interaction_'+key).value!==value)throw Error(scene+key);
}
elements.srairl_scenario.value='curve';elements.srairl_custom_speeds.checked=true;syncScenarioSpeeds();
if($('srairl_interaction_elcv_speed').value!==19)throw Error('custom value lost');
""".replace('DEFAULTS',__import__('json').dumps(SCENARIO_SPEED_CONFIG)).replace('FUNCTION',function)
    subprocess.run(['node','-e',script],check=True,capture_output=True,text=True)


def test_transition_distance_is_true_arc_length():
    from env.tasks.srairl_metadrive_world import _LaneTransition
    class Lane:
        def __init__(self,y):self.y=y
        def position(self,s,lateral=0):return np.array([s,self.y-lateral])
        def heading_theta_at(self,s):return 0.
    transition=_LaneTransition(Lane(0),0,Lane(3.5),80,length=80)
    assert transition.length>80
    positions=np.array([transition.position(s) for s in np.linspace(0,transition.length,1001)])
    assert np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()==pytest.approx(transition.length,rel=1e-4)
    assert abs(transition.heading_theta_at(0))<.001
    assert abs(transition.heading_theta_at(transition.length))<.001


def test_reference_normalization_matches_online_reward_observation():
    from datasets.srairl.native_windows import _normalize_joint
    from env.adapters.srairl_observation import SRAIRLObservationAdapter
    roles=('elcv','rfv','rlv_current','rlv_target')
    values=np.random.default_rng(7).uniform([-4,0,10,-1],[4,200,25,1],(11,4,4))
    adapter=SRAIRLObservationAdapter(history_len=11)
    online=np.stack([adapter.frame(dict(zip(roles,row))) for row in values])
    np.testing.assert_allclose(_normalize_joint(values),online,atol=1e-7)


def test_episode_split_rejects_duplicate_seed_in_same_scenario():
    from datasets.srairl.native_windows import assign_episode_splits
    with pytest.raises(ValueError,match='duplicate'):
        assign_episode_splits([dict(scenario='curve',seed=1)]*3)


def test_formal_calibration_rejects_single_axis_report_and_boolean_override():
    from env.adapters.srairl_calibration import action_calibration_passed
    report=dict(equivalence_status='passed',physics_evaluation_allowed=True,
        metrics={a:{s:dict(passed=True,mae_mps2=.01,max_error_mps2=.02,wrong_sign_count=0)
                    for s in ('calibration_grid','held_out')} for a in ('longitudinal','lateral')})
    assert action_calibration_passed(report)
    assert not action_calibration_passed(report,require_closed_loop=True)


def test_studio_catalog_serves_authoritative_speed_and_calibration(monkeypatch):
    import studio.server as server
    for name in ('model_catalog','srairl_model_catalog','reward_guide_model_catalog','srairl_pipeline_bundle_catalog','srairl_expert_catalog'):
        monkeypatch.setattr(server,name,lambda:[])
    handler=object.__new__(server.Handler)
    handler.path='/api/catalog'
    handler.respond=lambda value:value
    result=handler.get()
    assert result['srairl_scenario_speeds']==SCENARIO_SPEED_CONFIG
    assert isinstance(result['pipeline_calibration_passed'],bool)


@pytest.mark.parametrize('scene',['straight','curve','merge'])
def test_shared_reference_round_trip_matches_real_vehicle_world_state(scene):
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    env=SRAIRLProceduralEnv(scene,3,0.,seed=81,horizon=100)
    try:
        env.reset(seed=81)
        for role,state in env._roles_live().items():
            vehicle=env.world.role_vehicle(role)
            np.testing.assert_allclose(env.reference_to_world(role,state),
                np.concatenate((vehicle.position[:2],vehicle.velocity[:2])),atol=.03)
    finally:env.close()


def test_candidate_collision_gate_checks_vehicle_footprints():
    from rl.srairl.pipeline.selector import vehicle_footprints_overlap
    first=np.array([[0.,0.,20.,0.]])
    assert vehicle_footprints_overlap(first,np.array([[4.,0.,20.,0.]]),(4.5,1.8),(4.5,1.8))[0]
    assert not vehicle_footprints_overlap(first,np.array([[0.,3.5,20.,0.]]),(4.5,1.8),(4.5,1.8))[0]
    assert vehicle_footprints_overlap(first,np.array([[2.,1.,0.,20.]]),(4.5,1.8),(4.5,1.8))[0]


def test_evaluation_records_measured_jerk_and_control_saturation_once_per_frame():
    from rl.srairl.pipeline.evaluator import EpisodeMetrics
    metrics=EpisodeMetrics()
    for time,accel in ((.1,1.),(.2,2.)):
        info={'time':time,'dt_s':.1,'control_telemetry':{'elcv':{
            'desired_acceleration':[1.,0.],'metadrive_action':[1.,.2],
            'measured_acceleration':[accel,0.]}}}
        metrics.update_environment(info)
    metrics.update_environment(info)
    assert len(metrics.controls['elcv'])==2
    assert metrics.controls['elcv'][-1]['jerk_mps3']==[10.,0.]
    assert metrics.controls['elcv'][-1]['steering_saturated']
