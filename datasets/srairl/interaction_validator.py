"""Reject NGSIM role bindings that do not describe a lane-change interaction."""
import numpy as np


def _road_coordinates(states,heading):
    tangent=np.asarray([np.cos(heading),np.sin(heading)])
    normal=np.asarray([-tangent[1],tangent[0]])
    xy=np.asarray(states[:,1:3],dtype=float)
    return np.column_stack((xy@tangent,xy@normal))


def _state_at(track,time):
    states=np.asarray(track['states'],dtype=float)
    return np.asarray([np.interp(float(time),states[:,0],states[:,i]) for i in range(1,5)])


def validate_interaction_scenario(scenario, *, min_lane_change_frames=3, raise_on_error=True):
    roles={'elcv':scenario.elcv_id,'rfv':scenario.rfv_id,
           'rlv_current':scenario.rlv_current_id,'rlv_target':scenario.rlv_target_id}
    errors=[];tracks={name:scenario.tracks[ident] if ident is not None else None for name,ident in roles.items()}
    if any(tracks[name] is None for name in ('rlv_current','rlv_target')):
        errors.append('RLV_current and RLV_target roles are required')
    if errors:
        if raise_on_error:raise ValueError('invalid SR-AIRL interaction clip: '+'; '.join(errors))
        return {'valid':False,'errors':errors}
    elcv=np.asarray(tracks['elcv']['states'],dtype=float);rfv=np.asarray(tracks['rfv']['states'],dtype=float)
    headings=[];road_states={}
    for name,track in tracks.items():
        states=np.asarray(track['states'],dtype=float);moving=np.hypot(states[:,3],states[:,4])>.5
        angles=np.unwrap(np.arctan2(states[moving,4],states[moving,3]))
        if not len(angles):errors.append(f'{name} has no usable direction samples');continue
        heading=float(np.angle(np.mean(np.exp(1j*angles))))
        spread=float(np.max(np.abs(np.unwrap(angles-heading)))) if len(angles)>1 else 0.
        if spread>np.deg2rad(15.):errors.append(f'{name} turns more than 15 degrees; map-matched lane geometry is required')
        headings.append(heading);road_states[name]=(_road_coordinates(states,heading),states)
    if errors:
        if raise_on_error:raise ValueError('invalid SR-AIRL interaction clip: '+'; '.join(errors))
        return {'valid':False,'errors':errors}
    heading=float(np.angle(np.mean(np.exp(1j*np.asarray(headings)))))
    lateral_tracks={name:_road_coordinates(np.asarray(track['states'],dtype=float),heading)[:,1] for name,track in tracks.items()}
    origin=min(float(np.min(values)) for values in lateral_tracks.values())
    source=np.rint((lateral_tracks['elcv']-origin)/float(scenario.lane_width)).astype(int)
    changes=np.flatnonzero(source[1:]!=source[:-1])+1
    change=None
    if len(changes):
        for index in changes:
            target=int(source[index]);end=min(len(source),index+int(min_lane_change_frames))
            if end-index>=min_lane_change_frames and np.all(source[index:end]==target):change=int(index);break
    if change is None:errors.append('ELCV has no sustained lane change in the clip')
    else:
        target_lane=int(source[change]);source_lane=int(source[change-1]);t=float(elcv[change,0]);x=float(road_states['elcv'][0][change,0])
        frame=lambda track,t:np.asarray([np.interp(t,track[:,0],track[:,i]) for i in range(1,5)])
        for name in ('rfv','rlv_current','rlv_target'):
            states=np.asarray(tracks[name]['states'],dtype=float)
            if t<states[0,0]-1e-6 or t>states[-1,0]+1e-6:errors.append(f'{name} has no state at ELCV lane-change onset')
        if errors:
            if raise_on_error:raise ValueError('invalid SR-AIRL interaction clip: '+'; '.join(errors))
            return {'valid':False,'errors':errors,'lane_change_frame':change,'lane_change_time_s':t,
                    'source_lane':source_lane,'target_lane':target_lane,'road_heading_rad':heading}
        r=frame(rfv,t)
        q=np.asarray([r[0],r[1]]);r_s=float(q@np.asarray([np.cos(heading),np.sin(heading)]));r_l=float(q@np.asarray([-np.sin(heading),np.cos(heading)]))
        r_lane=int(round((r_l-origin)/scenario.lane_width))
        if r_lane!=target_lane:errors.append('RFV is not in ELCV target lane at lane-change onset')
        if not r_s<x:errors.append('RFV is not behind ELCV at lane-change onset')
        for name,expected_lane,relation in (('rlv_current',source_lane,'ahead'),('rlv_target',target_lane,'ahead')):
            track=np.asarray(tracks[name]['states'],dtype=float);q=frame(track,t)
            qpos=np.asarray([q[0],q[1]]);q_s=float(qpos@np.asarray([np.cos(heading),np.sin(heading)]));q_l=float(qpos@np.asarray([-np.sin(heading),np.cos(heading)]))
            lane_id=int(round((q_l-origin)/scenario.lane_width))
            if lane_id!=expected_lane:errors.append(f'{name} is not in the expected {"source" if name=="rlv_current" else "target"} lane')
            if relation=='ahead' and q_s<=x:errors.append(f'{name} is not ahead of ELCV at lane-change onset')
        if source_lane==target_lane:errors.append('ELCV source and target lanes are identical')
        if abs(source_lane-target_lane)!=1:errors.append('ELCV lane change is not to an adjacent lane')
    initial_state=None
    if change is not None and not errors:
        t=float(elcv[change,0]);tangent=np.asarray([np.cos(heading),np.sin(heading)])
        rows={name:_state_at(track,t) for name,track in tracks.items()}
        positions={name:float(np.asarray(row[:2])@tangent) for name,row in rows.items()}
        speeds={name:float(np.asarray(row[2:4])@tangent) for name,row in rows.items()}
        initial_state={'time_s':t,'elcv_rfv_gap_m':positions['elcv']-positions['rfv'],
            'rlv_current_gap_m':positions['rlv_current']-positions['elcv'],
            'rlv_target_gap_m':positions['rlv_target']-positions['elcv'],
            'elcv_speed_mps':speeds['elcv'],'rfv_speed_mps':speeds['rfv'],
            'rlv_current_speed_mps':speeds['rlv_current'],'rlv_target_speed_mps':speeds['rlv_target']}
    result={'valid':not errors,'errors':errors,'lane_change_frame':change,
            'lane_change_time_s':None if change is None else float(elcv[change,0]),
            'source_lane':None if change is None else int(source[change-1]),
            'target_lane':None if change is None else int(source[change]),
            'road_heading_rad':heading,'initial_state':initial_state}
    if errors and raise_on_error:raise ValueError('invalid SR-AIRL interaction clip: '+'; '.join(errors))
    return result


def extract_initial_state_distribution(scenario, *, samples=5, window_s=2.0):
    """Return deterministic pre-merge states from one validated training clip.

    Samples are temporal points from the same recorded interaction, not
    statistically independent vehicles or recording blocks.
    """
    if isinstance(samples,bool) or int(samples)<1 or not np.isfinite(window_s) or window_s<0:
        raise ValueError('distribution sample count/window must be positive')
    result=validate_interaction_scenario(scenario)
    tracks={'elcv':scenario.tracks[scenario.elcv_id],'rfv':scenario.tracks[scenario.rfv_id],
            'rlv_current':scenario.tracks[scenario.rlv_current_id],'rlv_target':scenario.tracks[scenario.rlv_target_id]}
    times=np.asarray(tracks['elcv']['states'],dtype=float)[:,0]
    onset=float(result['lane_change_time_s']);start=max(float(times[0]),onset-float(window_s))
    cadence=float(np.median(np.diff(times))) if len(times)>1 else .1
    prechange_end=onset-cadence
    if prechange_end<start-1e-9:return []
    candidates=np.linspace(start,prechange_end,int(samples)) if int(samples)>1 else np.asarray([prechange_end])
    heading=float(result['road_heading_rad']);tangent=np.asarray([np.cos(heading),np.sin(heading)])
    normal=np.asarray([-tangent[1],tangent[0]])
    all_lateral=[]
    for track in tracks.values():
        states=np.asarray(track['states'],dtype=float);xy=states[:,1:3]
        all_lateral.extend((xy@normal).tolist())
    lateral_origin=float(min(all_lateral));out=[]
    for t in candidates:
        if any(t<float(np.asarray(track['states'])[0,0]) or t>float(np.asarray(track['states'])[-1,0]) for track in tracks.values()):
            continue
        state={name:_state_at(track,t) for name,track in tracks.items()}
        longitudinal={name:float(np.asarray(row[:2])@tangent) for name,row in state.items()}
        lateral={name:float(np.asarray(row[:2])@normal) for name,row in state.items()}
        lane={name:int(round((value-lateral_origin)/float(scenario.lane_width))) for name,value in lateral.items()}
        if lane['rfv']!=lane['rlv_target'] or lane['rlv_current']!=lane['elcv']:
            continue
        if not (longitudinal['rfv']<longitudinal['elcv']<longitudinal['rlv_target'] and
                longitudinal['elcv']<longitudinal['rlv_current']):
            continue
        speeds={name:float(np.asarray(row[2:4])@tangent) for name,row in state.items()}
        if min(speeds.values())<=0:continue
        out.append({'time_s':float(t),'elcv_rfv_gap_m':longitudinal['elcv']-longitudinal['rfv'],
            'rlv_current_gap_m':longitudinal['rlv_current']-longitudinal['elcv'],
            'rlv_target_gap_m':longitudinal['rlv_target']-longitudinal['elcv'],
            'elcv_speed_mps':speeds['elcv'],'rfv_speed_mps':speeds['rfv'],
            'rlv_current_speed_mps':speeds['rlv_current'],'rlv_target_speed_mps':speeds['rlv_target'],
            'source_lane':lane['elcv'],'target_lane':lane['rfv']})
    if not out:
        raise ValueError('validated interaction clip has no valid pre-change initial-state samples')
    return out
