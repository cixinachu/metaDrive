"""Describe fixed four-vehicle NGSIM training interactions; never alter collection settings."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scripts.plot_ngsim_analysis import load_training_clips,ego_states

ROLES=('elcv','rfv','rlv_current','rlv_target')


def duration_proxy(states,lanes,source,target,cross_index):
    """10–90% lateral excursion between observed stable endpoint plateaus."""
    if not (np.all(lanes[:10]==source) and np.all(lanes[-10:]==target)):
        return None,'endpoints_not_in_source_and_target_lanes'
    if np.median(np.abs(states[:10,4]))>.2 or np.median(np.abs(states[-10:,4]))>.2:
        return None,'endpoint_lateral_velocity_not_stable'
    y=savgol_filter(states[:,2],11,2)
    start=float(np.median(y[:10]));end=float(np.median(y[-10:]));delta=end-start
    if abs(delta)<1.:return None,'lateral_displacement_under_1m'
    progress=(y-start)/delta
    before=np.flatnonzero((progress[:-1]<.1)&(progress[1:]>=.1))
    before=before[before<cross_index]
    after=np.flatnonzero((progress[:-1]<.9)&(progress[1:]>=.9))
    after=after[after>=cross_index-1]
    if not len(before) or not len(after):return None,'10_90_crossings_not_observed'
    def crossing(i,threshold):
        return float(states[i,0]+(threshold-progress[i])/(progress[i+1]-progress[i])*(states[i+1,0]-states[i,0]))
    start_time=crossing(int(before[-1]),.1);end_time=crossing(int(after[0]),.9)
    if end_time<=start_time:return None,'nonpositive_duration'
    return dict(lateral_10_time_s=start_time,lateral_90_time_s=end_time,
        lateral_10_90_duration_s=end_time-start_time,available_future_after_90_s=float(states[-1,0]-end_time)),None


def analyze_event(selection,clip,manifest,raw):
    ego,states=ego_states(clip)
    start_ms=int(round(clip['source_start_time']*1000))
    times=start_ms+np.rint(states[:,0]*1000).astype(np.int64)
    ego_id=int(ego.get('source_id',ego['id'].split(':')[0]))
    output=dict(clip_id=manifest['id'],recording=selection['recording'],elcv_source_id=ego_id,
        accepted=False,rejection_reason='',duration_rejection_reason='')
    def reject(reason):output['rejection_reason']=reason;return output
    try:lanes=np.asarray([raw[(ego_id,int(stamp))][0] for stamp in times])
    except KeyError:return reject('missing_original_ego_lane_id')
    changes=[]
    for i in np.flatnonzero(lanes[1:]!=lanes[:-1])+1:
        if i>=10 and i+10<=len(lanes) and np.all(lanes[i-10:i]==lanes[i-1]) and np.all(lanes[i:i+10]==lanes[i]):
            if abs(lanes[i]-lanes[i-1])==1:changes.append(int(i))
    if not changes:return reject('no_1s_stable_adjacent_lane_transition')
    if len(changes)>1:return reject('multiple_stable_lane_transitions')
    i=changes[0];initial=i-20
    if initial<0:return reject('less_than_2s_before_lane_crossing')
    source=int(lanes[i-1]);target=int(lanes[i]);stamp=int(times[initial])
    if lanes[initial]!=source:return reject('initial_ego_not_in_source_lane')
    output.update(source_lane=source,target_lane=target,cross_time_ms=int(times[i]),
        initial_time_ms=stamp,initial_time_in_clip_s=float(states[initial,0]))
    duration,error=duration_proxy(states,lanes,source,target,i)
    output['duration_rejection_reason']=error or ''
    if duration:output.update(duration)
    ego_x=float(states[initial,1]);candidates={'rfv':[],'rlv_current':[],'rlv_target':[]}
    tracks={str(t['id']):t for t in clip['tracks']}
    for track in clip['tracks']:
        if track.get('ego'):continue
        s=np.asarray(track['states'],dtype=float)
        time=float(states[initial,0])
        if time<s[0,0]-1e-8 or time>s[-1,0]+1e-8:continue
        ident=int(track.get('source_id',track['id'].split(':')[0]))
        sample=raw.get((ident,stamp))
        if sample is None:continue
        lane=int(sample[0]);x=float(np.interp(time,s[:,0],s[:,1]));dx=x-ego_x
        if lane==target and dx<0:candidates['rfv'].append((-dx,str(track['id'])))
        if lane==target and dx>0:candidates['rlv_target'].append((dx,str(track['id'])))
        if lane==source and dx>0:candidates['rlv_current'].append((dx,str(track['id'])))
    missing=[role for role,values in candidates.items() if not values]
    if missing:return reject('missing_'+ '_and_'.join(missing))
    bindings={'elcv':str(ego['id']),**{role:min(values)[1] for role,values in candidates.items()}}
    time=float(states[initial,0]);end_time=float(states[min(i+10,len(states)-1),0])
    role_states={};lengths={}
    for role,track_id in bindings.items():
        track=tracks[track_id];s=np.asarray(track['states'],dtype=float)
        ident=int(track.get('source_id',track_id.split(':')[0]))
        output[role+'_source_id']=ident;output[role+'_track_id']=track_id
        if int(raw[(ident,stamp)][1])!=2:return reject('nearest_role_not_passenger_car_'+role)
        if s[0,0]>time+1e-8 or s[-1,0]<end_time-1e-8:return reject('fixed_role_missing_through_crossing_'+role)
        check_stamps=times[initial:min(i+11,len(times))]
        samples=[raw.get((ident,int(value))) for value in check_stamps]
        if any(value is None for value in samples):return reject('fixed_role_has_missing_original_frames_'+role)
        expected=source if role=='rlv_current' else target
        if role!='elcv' and any(int(value[0])!=expected for value in samples):
            return reject('neighbor_changes_lane_'+role)
        role_states[role]=np.asarray([np.interp(time,s[:,0],s[:,j]) for j in range(1,5)])
        lengths[role]=float(track['length']);output[role+'_length_m']=lengths[role]
        speed=float(role_states[role][2]);output[role+'_speed_mps']=speed
        if not np.isfinite(speed) or speed<=0:return reject('nonpositive_or_invalid_speed_'+role)
    positions={role:float(s[0]) for role,s in role_states.items()}
    speeds={role:float(s[2]) for role,s in role_states.items()}
    gaps={'rear':positions['elcv']-positions['rfv'],
        'current_front':positions['rlv_current']-positions['elcv'],
        'target_front':positions['rlv_target']-positions['elcv']}
    pairs={'rear':('rfv','elcv'),'current_front':('elcv','rlv_current'),'target_front':('elcv','rlv_target')}
    for key,center in gaps.items():
        follower,leader=pairs[key];clearance=center-(lengths[follower]+lengths[leader])/2
        dv=speeds[follower]-speeds[leader]
        output[key+'_center_gap_m']=center;output[key+'_clearance_m']=clearance
        output[key+'_closing_speed_mps']=dv;output[key+'_headway_s']=clearance/speeds[follower]
        output[key+'_ttc_s']=clearance/dv if dv>1e-6 and clearance>0 else None
        if clearance<=0:return reject('nonpositive_initial_longitudinal_clearance_'+key)
    crossing_time=float(states[i,0])
    crossing_positions={}
    for role,track_id in bindings.items():
        trajectory=np.asarray(tracks[track_id]['states'],dtype=float)
        crossing_positions[role]=float(np.interp(crossing_time,trajectory[:,0],trajectory[:,1]))
    if not (crossing_positions['rfv']<crossing_positions['elcv']<crossing_positions['rlv_target']
            and crossing_positions['elcv']<crossing_positions['rlv_current']):
        return reject('bound_role_order_changes_by_lane_crossing')
    output['target_slot_center_span_m']=positions['rlv_target']-positions['rfv']
    output['template_min_center_gap_6m']=min(gaps.values())>=6.
    output['template_speed_domain_0_25mps']=all(0<v<=25 for v in speeds.values())
    output['accepted']=True
    return output


def describe(frame,columns):
    result={}
    for name in columns:
        values=pd.to_numeric(frame.get(name,pd.Series(dtype=float)),errors='coerce').dropna().to_numpy()
        values=values[np.isfinite(values)]
        result[name]=dict(n=len(values),**({
            'min':float(values.min()),'p10':float(np.percentile(values,10)),
            'p25':float(np.percentile(values,25)),'p50':float(np.median(values)),
            'p75':float(np.percentile(values,75)),'p90':float(np.percentile(values,90)),
            'max':float(values.max()),'mean':float(values.mean())} if len(values) else {}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'outputs/ngsim_interaction_statistics'/(
        datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    selections=[ROOT/f'datasets/experiments/ngsim-mainline-v{i}/selected.json' for i in (1,2)]
    clips=load_training_clips(selections)
    events=[item for item in clips if item[0]['event_class']=='ego_lane_change']
    origins={int(item[0]['origin']) for item in events}
    locations={item[0]['location'] for item in events}
    if len(origins)!=1 or locations!={'i-80'}:raise ValueError('This protocol expects one I-80 training recording')
    needed_times=set()
    for _,clip,_ in events:
        _,s=ego_states(clip);needed_times.update((int(round(clip['source_start_time']*1000))+np.rint(s[:,0]*1000).astype(int)).tolist())
    raw_root=ROOT/'datasets/raw/ngsim/full'
    source=json.loads((raw_root/'manifest.json').read_text())
    pages=[page for page in source['pages'] if page['path'].startswith('i-80_')]
    raw={};conflicts=set();snapshots=defaultdict(dict);page_hashes={};duplicate_rows=0
    usecols=['vehicle_id','frame_id','global_time','lane_id','v_class','local_y','v_length','v_vel']
    for index,page in enumerate(pages,1):
        path=raw_root/'pages'/page['path']
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=page['sha256']:raise ValueError('Raw page hash mismatch: '+str(path))
        page_hashes[page['path']]=digest
        df=pd.read_csv(path,usecols=usecols)
        df=df[(df.global_time-df.frame_id*100).isin(origins)]
        for row in df.itertuples(index=False):
            ident=int(row.vehicle_id);stamp=int(row.global_time);lane=int(row.lane_id)
            if stamp in needed_times:
                key=(ident,stamp);value=(lane,int(row.v_class))
                if key in raw:
                    duplicate_rows+=1
                    if raw[key]!=value:conflicts.add(key)
                else:raw[key]=value
            if int(row.frame_id)%10==0:
                snapshot=snapshots[stamp]
                center=(float(row.local_y)-float(row.v_length)/2)*.3048
                if 1<=lane<=6 and 100.<=center<300.:
                    snapshot[ident]=float(row.v_vel)*.3048
        if index%10==0 or index==len(pages):print(f'原始 Lane_ID 核对：{index}/{len(pages)} 页',flush=True)
    for key in conflicts:raw.pop(key,None)
    print(f'分析 {len(events)} 个训练换道片段；原始冲突键 {len(conflicts)}',flush=True)
    results=[];seen=set();duplicates=[]
    for selection,clip,meta in events:
        result=analyze_event(selection,clip,meta,raw)
        if 'cross_time_ms' in result:
            key=(result['recording'],result['elcv_source_id'],result['cross_time_ms'])
            if key in seen:
                result.update(accepted=False,rejection_reason='duplicate_physical_event');duplicates.append(key)
            seen.add(key)
        results.append(result)
    all_events=pd.DataFrame(results)
    accepted=all_events[all_events.accepted].copy()
    all_events.to_csv(output/'all_event_audit.csv',index=False)
    accepted.to_csv(output/'accepted_initial_states.csv',index=False)
    density=pd.DataFrame([dict(time_ms=t,vehicles=len(vehicles),density_veh_per_km_per_lane=len(vehicles)/(.2*6),
        mean_reported_speed_mps=float(np.mean(list(vehicles.values()))) if vehicles else np.nan)
        for t,vehicles in sorted(snapshots.items())])
    density.to_csv(output/'training_recording_density_1s.csv',index=False)
    metrics=[role+'_speed_mps' for role in ROLES]
    metrics += [key+'_'+suffix for suffix in ('center_gap_m','clearance_m','closing_speed_mps','headway_s','ttc_s')
        for key in ('rear','current_front','target_front')]
    metrics+=['target_slot_center_span_m','lateral_10_90_duration_s','available_future_after_90_s']
    stats=describe(accepted,metrics)
    summary=dict(protocol='NGSIM fixed-role initial-state descriptive statistics v2',
        selected_training_clips=len(clips),lane_change_candidates=len(events),accepted_fixed_role_events=len(accepted),
        recording_groups=sorted({item[0]['recording'] for item in events}),
        rejected_reasons=dict(Counter(all_events.loc[~all_events.accepted,'rejection_reason'])),
        duration_unavailable_reasons=dict(Counter(accepted.loc[accepted.duration_rejection_reason!='','duration_rejection_reason'])) if len(accepted) else {},
        duplicate_event_count=len(duplicates),raw_lane_conflicts=len(conflicts),raw_duplicate_rows=duplicate_rows,
        statistics=stats,density_statistics=describe(density,['density_veh_per_km_per_lane','mean_reported_speed_mps']),
        basic_speed_gap_domain_events=int((accepted.template_min_center_gap_6m & accepted.template_speed_domain_0_25mps).sum()) if len(accepted) else 0,
        selection_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selections},
        source_manifest_sha256=hashlib.sha256((raw_root/'manifest.json').read_bytes()).hexdigest(),raw_page_sha256=page_hashes,
        definitions=dict(initial_state='2.0 s before original Lane_ID transition, not physical maneuver onset',
            stable_lane_transition='Adjacent original Lane_ID, at least 1.0 s stable before and after',
            fixed_roles='Nearest source leader, nearest target leader/follower at initial state; same identities and neighbor lanes required through 1 s after crossing; follower/leader longitudinal order must still hold at lane crossing',
            vehicle_classes='ELCV and all three nearest roles must be passenger cars (v_class=2); trucks are not skipped to pick a more distant car',
            longitudinal_position='Vehicle center in metres from processed clips; original importer subtracts half length from front reference',
            duration='Diagnostic 10–90% lateral excursion between stable observed endpoint plateaus; not full maneuver duration',
            density='One-second snapshots over full training recording, fixed raw longitudinal center ROI [100,300) m, all six mainline lanes and all vehicle classes',
            ttc='Longitudinal clearance / positive follower-minus-leader speed; non-closing states are missing, not zero'),
        density_roi_coverage_verified=False,
        limitations=['Basic speed/gap compatibility only checks positive speeds up to 25 m/s and center gaps >= 6 m; it does not certify controller feasibility.',
            'ROI observation coverage has not been independently verified; zero counts mean no recorded vehicles, not verified empty road.',
            'One selected I-80 training recording; event-selected clips are not unbiased full NGSIM.',
            'Different events and adjacent density snapshots share traffic and are not statistically independent.',
            'Only initially positive longitudinal clearances and positive speeds are accepted; rejected events are retained in audit.',
            'TTC is a longitudinal pair diagnostic, not collision prediction across lateral maneuvers.',
            'Density is descriptive observed traffic; it is not MetaDrive traffic_density in [0,1].',
            'No expert collector parameters were changed or automatically fitted.'])
    summary['analysis_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['plot_code_sha256']=hashlib.sha256((ROOT/'scripts/plot_ngsim_interaction_statistics.py').read_bytes()).hexdigest()
    pd.DataFrame.from_dict(stats,orient='index').rename_axis('metric').to_csv(output/'parameter_quantiles.csv')
    joint=[role+'_speed_mps' for role in ROLES]+[key+'_center_gap_m' for key in ('rear','current_front','target_front')]
    accepted.reindex(columns=joint).corr().to_csv(output/'joint_initial_state_correlations.csv')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    from scripts.plot_ngsim_interaction_statistics import plot_statistics
    plot_statistics(accepted,density,output)
    print(json.dumps(dict(output=str(output),candidates=len(events),accepted=len(accepted),statistics=stats,
        rejected_reasons=summary['rejected_reasons'],density_statistics=summary['density_statistics']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
