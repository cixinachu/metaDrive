from studio.worker import action_calibration_passed, srairl_experiment_protocol
from tasks.registry import TASK_REGISTRY


def test_calibration_gate_requires_grid_and_held_out_for_both_axes():
    summary={'equivalence_status':'passed','metrics':{
        axis:{split:{'passed':True} for split in ('calibration_grid','held_out')}
        for axis in ('longitudinal','lateral')}}
    assert action_calibration_passed(summary)
    summary['metrics']['lateral']['held_out']['passed']=False
    assert not action_calibration_passed(summary)
    assert not action_calibration_passed({'equivalence_status':'passed','metrics':{}})


def test_srairl_experiment_protocols_keep_reproduction_sensitivity_and_transfer_separate():
    assert srairl_experiment_protocol({'scene_source':'ngsim','dynamics_mode':'legacy_kinematic'})['id']=='A'
    assert srairl_experiment_protocol({'scene_source':'ngsim','dynamics_mode':'legacy_kinematic',
        'srairl_reactive_background':True})['qualification']=='sensitivity_only'
    assert srairl_experiment_protocol({'scene_source':'procedural','dynamics_mode':'metadrive_physics',
        'scenario':'straight'},True)['qualification']=='quantitative_transfer'
    assert srairl_experiment_protocol({'scene_source':'procedural','dynamics_mode':'metadrive_physics',
        'scenario':'merge'},False)['qualification']=='exploratory_only'
    assert srairl_experiment_protocol({'scene_source':'ngsim','dynamics_mode':'metadrive_physics'})['id']=='custom'


def test_candidate_evaluation_remains_partial_until_model_bundle_and_full_map_gates_pass():
    evaluation=TASK_REGISTRY['evaluation']
    assert evaluation.status=='partial'
    assert 'CandidateBundle' in evaluation.description
    assert 'full episodes pass' in evaluation.description
