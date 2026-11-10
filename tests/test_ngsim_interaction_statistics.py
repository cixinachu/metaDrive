"""Verify fixed-role statistics against known trajectories and original lane labels."""
import copy
import numpy as np
import pytest
from scripts.stat_ngsim_interactions import analyze_event


@pytest.fixture
def interaction():
    t=np.round(np.arange(201)*.1,1);progress=np.clip((t-6)/8,0,1)
    ego_x=30+10*t
    tracks=[]
    for ident,x,y,vx in [(1,ego_x,3.6576*(1-progress),10),
            (2,ego_x-36+2*t,np.zeros_like(t),12),
            (3,ego_x+43-t,np.full_like(t,3.6576),9),
            (4,ego_x+25,np.zeros_like(t),10)]:
        vy=np.gradient(y,t)
        tracks.append(dict(id=str(ident)+':0',source_id=str(ident),ego=ident==1,length=4.5,
            states=np.column_stack((t,x,y,np.full_like(t,vx),vy)).tolist()))
    raw={}
    for stamp,time in zip(np.rint(t*1000).astype(int),t):
        for ident in (1,2,3,4):
            lane=(1 if time<10 else 2) if ident==1 else (1 if ident==3 else 2)
            raw[(ident,int(stamp))]=(lane,2)
    return dict(recording='synthetic:0'),dict(source_start_time=0,tracks=tracks),dict(id='test'),raw


def test_fixed_role_gaps_relative_speed_and_duration(interaction):
    result=analyze_event(*interaction)
    assert result['accepted'],result
    assert result['initial_time_ms']==8000 and result['cross_time_ms']==10000
    assert result['rear_center_gap_m']==pytest.approx(20)
    assert result['rear_clearance_m']==pytest.approx(15.5)
    assert result['current_front_center_gap_m']==pytest.approx(35)
    assert result['target_front_center_gap_m']==pytest.approx(25)
    assert result['rear_closing_speed_mps']==pytest.approx(2)
    assert result['current_front_closing_speed_mps']==pytest.approx(1)
    assert result['rear_ttc_s']==pytest.approx(7.75)
    assert result['target_front_ttc_s'] is None
    assert result['lateral_10_90_duration_s']==pytest.approx(6.4,abs=.02)
    assert result['rfv_source_id']==2 and result['rlv_current_source_id']==3


def test_nearest_truck_is_rejected_instead_of_skipped(interaction):
    selection,clip,meta,raw=copy.deepcopy(interaction)
    raw[(3,8000)]=(1,3)
    result=analyze_event(selection,clip,meta,raw)
    assert not result['accepted']
    assert result['rejection_reason']=='nearest_role_not_passenger_car_rlv_current'


def test_role_identity_and_lane_must_remain_fixed(interaction):
    selection,clip,meta,raw=copy.deepcopy(interaction)
    raw[(2,10500)]=(3,2)
    result=analyze_event(selection,clip,meta,raw)
    assert not result['accepted']
    assert result['rejection_reason']=='neighbor_changes_lane_rfv'


def test_initial_target_leader_must_still_be_ahead_at_crossing(interaction):
    selection,clip,meta,raw=copy.deepcopy(interaction)
    track=next(track for track in clip['tracks'] if track['source_id']=='4')
    s=np.asarray(track['states']);s[:,1]=75+5*s[:,0];s[:,3]=5
    track['states']=s.tolist()
    result=analyze_event(selection,clip,meta,raw)
    assert not result['accepted']
    assert result['rejection_reason']=='bound_role_order_changes_by_lane_crossing'
