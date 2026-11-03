import numpy as np
import pytest

from env.tasks.srairl_trajectory_tracker import (
    SRAIRLTrajectoryTracker,
    TrackerGains,
    trajectory_tracking_metrics,
)


def test_tracker_acceleration_and_tracking_metrics_validate_lane_state_contract():
    tracker=SRAIRLTrajectoryTracker(None)
    actual=np.asarray([[0.,0.,10.,0.],[1.,.5,11.,0.]])
    reference=np.asarray([[0.,0.,10.,0.],[.8,.3,10.,0.]])
    metrics=trajectory_tracking_metrics(actual,reference)
    assert metrics==pytest.approx({
        'longitudinal_rmse_m':.2/2**.5,
        'lateral_rmse_m':.2/2**.5,
        'speed_rmse_mps':1./2**.5,
    })
    assert tracker.desired_acceleration(actual[1],reference[1]).shape==(2,)
    with pytest.raises(ValueError,match='matching non-empty'):
        trajectory_tracking_metrics(actual,reference[:1])


def test_meta_drive_tracker_tracks_straight_accelerating_and_lane_change_references():
    from env.tasks.srairl_interaction import SRAIRLInteractionScenario
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv

    gains=TrackerGains()
    interaction=SRAIRLInteractionScenario(rfv_x=15.,rfv_speed=10.,rlv_current_x=200.,
        rlv_target_x=100.,interaction_end_x=300.)
    env=SRAIRLProceduralEnv('straight',2,0.,seed=311,horizon=100,rfv_response='fixed_speed',
        interaction_scenario=interaction)
    try:
        tracker=SRAIRLTrajectoryTracker(env.action_adapter,gains)
        for case in ('constant_speed','constant_acceleration','smooth_lane_change'):
            env.reset(seed=311)
            vehicle=env.world.role_vehicle('elcv');lane=env._source_lane
            initial=tracker.lane_state(vehicle,lane);s0=float(initial[0]);v0=float(initial[2])
            if case=='smooth_lane_change':
                lane_s,_=lane.local_coordinates(vehicle.position[:2])
                target_d=-float(lane.local_coordinates(env._target_lane.position(lane_s,0.)[:2])[1])
            actual=[];reference=[];info={}
            step_limit=100 if case=='smooth_lane_change' else 20
            for step in range(step_limit):
                t=(step+1)*env.dt
                if case=='constant_speed':
                    target=np.asarray([s0+v0*t,0.,v0,0.])
                elif case=='constant_acceleration':
                    accel=.8
                    target=np.asarray([s0+v0*t+.5*accel*t*t,0.,v0+accel*t,0.])
                else:
                    phase=min(t/3.5,1.)
                    blend=10*phase**3-15*phase**4+6*phase**5
                    lateral=target_d*blend
                    lateral_speed=(target_d*(30*phase**2-60*phase**3+30*phase**4)/3.5) if phase<1. else 0.
                    target=np.asarray([s0+v0*t,lateral,v0,lateral_speed])
                action=tracker.desired_acceleration(tracker.lane_state(vehicle,lane),target)
                _,info=env.step({'elcv':action,'rfv':np.zeros(2)})
                actual.append(tracker.lane_state(vehicle,lane));reference.append(target)
                if case=='smooth_lane_change' and info['lane_change_completed']:
                    break
                if info['done']:
                    break
            metrics=trajectory_tracking_metrics(actual,reference)
            assert metrics['longitudinal_rmse_m']<1.5, (case,metrics)
            assert metrics['lateral_rmse_m']<.7, (case,metrics)
            assert metrics['speed_rmse_mps']<.8, (case,metrics)
            if case=='smooth_lane_change':
                assert info['lane_change_completed']
                assert info['merge_success']
                assert not info['metadrive_primary_collision']
                assert not info['metadrive_out_of_road']
    finally:
        env.close()
