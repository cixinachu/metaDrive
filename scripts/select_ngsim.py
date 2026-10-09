"""Audit complete NGSIM highways, select versioned windows and build MetaDrive clips."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,collections,hashlib,json
import numpy as np
import pandas as pd
from datasets.importer import ROOT,digest,STORE
CONFIG=ROOT/'configs/datasets/ngsim_selection_v1.json'
OUT=ROOT/'datasets/experiments/ngsim-mainline-v1'
COLS='vehicle_id frame_id global_time local_x local_y v_length v_width v_class lane_id preceding v_vel v_acc'.split()

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def stable_changes(lanes,n=10):
    return [i for i in np.flatnonzero(np.diff(lanes)!=0)+1 if i>=n and i+n<=len(lanes) and abs(lanes[i]-lanes[i-1])==1 and np.all(lanes[i-n:i]==lanes[i-1]) and np.all(lanes[i:i+n]==lanes[i])]

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    if not cfg['version'].replace('-','').replace('_','').isalnum():raise ValueError('Invalid protocol version')
    OUT=ROOT/'datasets/experiments'/cfg['version']
    if cfg['duration_seconds']!=20 or cfg['sample_interval_ms']!=100:raise ValueError('v1 implements 20s clips at 10Hz; a new duration requires a new protocol implementation')
    OUT.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'datasets/raw/ngsim/full';source=json.loads((raw/'manifest.json').read_text())
    if digest(raw/'trajectories.csv')!=source['sha256']:raise ValueError('Raw CSV hash mismatch')
    config_sha=digest(args.config);script_sha=digest(Path(__file__));audit=[];candidates=[];recordings={};rejections=collections.Counter()
    for loc,lanes in cfg['mainline_lanes'].items():
        page_meta=[m for m in source['pages'] if m['path'].startswith(loc+'_')]
        files=[raw/'pages'/m['path'] for m in page_meta]
        for f,m in zip(files,page_meta):
            if digest(f)!=m['sha256']:raise ValueError(f'Source page hash mismatch: {f.name}')
        frames=[pd.read_csv(f) for f in files]
        df=pd.concat(frames,ignore_index=True);del frames
        if len(df)!=source['counts'][loc]:raise ValueError('Missing source pages')
        before=len(df);df=df.drop_duplicates();duplicates=before-len(df)
        n_before_projection=len(df);df=df[COLS].drop_duplicates();projection_duplicates=n_before_projection-len(df)
        df['origin']=df.global_time-df.frame_id*100
        for origin,g in df.groupby('origin',sort=True):
            origin=int(origin);split=next((s for s,v in cfg['splits'].items() if v['location']==loc and origin in v['origins']),None)
            if split is None:raise ValueError(f'Unspecified recording {loc}/{origin}')
            g=g.sort_values(['vehicle_id','global_time']).copy()
            conflict=g.duplicated(['vehicle_id','global_time'],keep=False)
            bad_ids=set(g.loc[conflict,'vehicle_id']);g=g[~g.vehicle_id.isin(bad_ids)]
            key=f'{loc}:{origin}';recordings[key]=(g,lanes,split)
            leader_index=g.set_index(['vehicle_id','global_time'])
            by_id={int(i):r.set_index('global_time',drop=False) for i,r in g.groupby('vehicle_id')}
            grouped=g.groupby('vehicle_id')
            dt=grouped.global_time.diff()
            dx=(grouped.local_y.diff()-grouped.v_length.diff()/2)*.3048/.1
            dy=grouped.local_x.diff()*.3048/.1
            bad=(dt==100)&g.lane_id.between(1,lanes)&((dx < cfg['longitudinal_velocity_m_s'][0])|(dx > cfg['longitudinal_velocity_m_s'][1])|(dy.abs()>cfg['lateral_velocity_abs_m_s']))
            bad_times=np.sort(g.loc[bad,'global_time'].unique())
            at_time={int(t):r for t,r in g.groupby('global_time')}

            stats=collections.Counter();stats['conflicting_tracks']=len(bad_ids)
            for ego,tr in by_id.items():
                stats['vehicles']+=1
                if not (tr.v_class==cfg['ego_vehicle_class']).all():stats['not_passenger_car']+=1;continue
                valid=(tr.lane_id.between(1,lanes)&(tr.v_length*.3048).between(*cfg['length_m'])&(tr.v_width*.3048).between(*cfg['width_m'])).to_numpy()
                # Segment at every missing frame, lane-domain exit or invalid dimensions.
                tt=tr.global_time.to_numpy();cuts=np.flatnonzero((np.diff(tt)!=100)|(~valid[:-1])|(~valid[1:]))+1
                parts=[a for a in np.split(np.arange(len(tr)),cuts) if len(a)>=201 and valid[a].all()]
                if not parts:stats['insufficient_continuous_20s']+=1;continue
                events=[]
                for part in parts:
                    ls=tr.iloc[part].lane_id.to_numpy()
                    for i in stable_changes(ls,cfg['stable_lane_frames']):events.append((int(tt[part[i]]),part,i))
                if events:
                    _,part,event=min(events,key=lambda z:z[0]);begin=min(max(event-100,0),len(part)-201)
                else:
                    part=max(parts,key=lambda a:(len(a),-int(tt[a[0]])));begin=(len(part)-201)//2
                w=tr.iloc[part[begin:begin+201]];t=w.global_time.to_numpy();x=(w.local_y-w.v_length/2).to_numpy()*.3048;y=w.local_x.to_numpy()*.3048
                vx=np.gradient(x,.1);vy=np.gradient(y,.1)
                if not np.isfinite(np.c_[x,y,vx,vy]).all():stats['nonfinite']+=1;continue
                if vx.min()<cfg['longitudinal_velocity_m_s'][0] or vx.max()>cfg['longitudinal_velocity_m_s'][1] or np.abs(vy).max()>cfg['lateral_velocity_abs_m_s']:stats['position_jump_velocity']+=1;continue
                half=w.v_width.to_numpy()*.3048/2
                if np.any(y-half<0) or np.any(y+half>lanes*cfg['lane_width_m']):stats['outside_approximate_road']+=1;continue
                # Start must fit inside the approximate road; future low TTC is kept.
                lane=w.lane_id.to_numpy();leader=w.preceding.to_numpy().astype(int)
                event_class='ego_lane_change' if stable_changes(lane,cfg['stable_lane_frames']) else 'other_mainline'
                refs=leader_index.reindex(pd.MultiIndex.from_arrays([leader,t]))
                gaps=(refs.local_y.to_numpy()-refs.v_length.to_numpy()-w.local_y.to_numpy())*.3048
                same=(refs.lane_id.to_numpy()==lane)&(gaps>0)
                closing=(w.v_vel.to_numpy()-refs.v_vel.to_numpy())*.3048
                valid_ttc=same&(closing>.1)
                min_ttc=float(np.min(gaps[valid_ttc]/closing[valid_ttc])) if valid_ttc.any() else None
                longest=0;run=0;prev=None;cutin=False
                for ok,la,lead in zip(same,lane,leader):
                    pair=(la,lead);run=run+1 if ok and pair==prev else (1 if ok else 0)
                    prev=pair if ok else None;longest=max(longest,run)
                changes=np.flatnonzero(np.diff(leader)!=0)+1
                for j in changes:
                    if j<10 or not same[j] or gaps[j]>cfg['cut_in_max_gap_m'] or not np.all(lane[j-10:j+1]==lane[j]):continue
                    stamp=int(t[j]);leadtrack=by_id.get(int(leader[j]))
                    if leadtrack is None or stamp-1000 not in leadtrack.index:continue
                    old=int(leadtrack.loc[stamp-1000].lane_id)
                    future=leadtrack.reindex(np.arange(stamp,stamp+1000,100)).lane_id
                    if 1<=old<=lanes and abs(old-int(lane[j]))==1 and (future==lane[j]).all():cutin=True
                if event_class!='ego_lane_change':
                    event_class='cut_in' if cutin else ('car_following' if (longest-1)*.1>cfg['car_following_min_seconds'] else 'other_mainline')
                identity=f'{cfg["version"]}:{config_sha}:{script_sha}:{key}:{ego}:{int(t[0])}'
                candidate=dict(candidate_id=hashlib.sha256(identity.encode()).hexdigest()[:24],recording=key,location=loc,origin=origin,split=split,ego_id=ego,start_ms=int(t[0]),end_ms=int(t[-1]),event_class=event_class,min_recorded_ttc_s=min_ttc,risk='ttc_lt_3' if min_ttc is not None and min_ttc<cfg['risk_ttc_seconds'] else 'other_or_unobserved',following_seconds=(longest-1)*.1,selected=False,clip_id='',rejection='')
                # Quality audit every candidate before sampling, independent of algorithm outcomes.
                k=np.searchsorted(bad_times,int(t[0]),side='right')
                if k<len(bad_times) and bad_times[k]<=int(t[-1]):
                    candidate['rejection']='background_position_jump'
                initial=at_time[int(t[0])]
                neighbors=initial[(initial.vehicle_id!=ego)&initial.lane_id.between(1,lanes)]
                nx=(neighbors.local_y-neighbors.v_length/2).to_numpy()*.3048
                ny=neighbors.local_x.to_numpy()*.3048
                overlaps=(np.abs(nx-x[0])<(neighbors.v_length.to_numpy()*.3048+float(w.iloc[0].v_length)*.3048)/2)&(np.abs(ny-y[0])<(neighbors.v_width.to_numpy()*.3048+float(w.iloc[0].v_width)*.3048)/2)
                if overlaps.any():candidate['rejection']='initial_ego_overlap'
                candidates.append(candidate);stats['accepted_candidates' if not candidate['rejection'] else candidate['rejection']]+=1
            audit.append(dict(recording=key,split=split,conflicting_vehicle_ids=sorted(int(i) for i in bad_ids),unique_rows=len(g),first_ms=int(g.global_time.min()),last_ms=int(g.global_time.max()),counts=dict(stats)))
            print('screened',key,dict(stats),flush=True)
        audit.append(dict(location=loc,raw_rows=before,exact_duplicates_removed=duplicates,projection_duplicates_removed=projection_duplicates))
        del df
    # Per-class uniform deterministic sample; keep rejected materializations in ledger.
    for split in cfg['splits']:
        for cls in cfg['event_classes']:
            pool=[c for c in candidates if c['split']==split and c['event_class']==cls and not c['rejection']]
            pool.sort(key=lambda c:hashlib.sha256(f'{cfg["seed"]}:{c["candidate_id"]}'.encode()).hexdigest())
            selected=0
            for c in pool:
                if selected>=cfg['max_materialized_per_split_and_class']:break
                g,lanes,_=recordings[c['recording']]
                try:
                    identifier=materialize(c,g,lanes,cfg,source,config_sha)
                except ValueError as exc:
                    c['rejection']=str(exc);rejections[str(exc)]+=1;continue
                c.update(selected=True,clip_id=identifier);selected+=1
            print('selected',split,cls,selected,'/',len(pool),flush=True)
    pd.DataFrame(candidates).to_csv(OUT/'candidates.csv',index=False)
    chosen=[c for c in candidates if c['selected']]
    summary=dict(version=cfg['version'],config_sha256=config_sha,selection_code_sha256=script_sha,source_sha256=source['sha256'],downloaded_rows=source['rows'],screened_highway_rows=sum(source['counts'][s] for s in cfg['mainline_lanes']),audit=audit,candidates=len(candidates),quality_eligible=sum(not c['rejection'] for c in candidates),selected=len(chosen),materialization_rejections=dict(rejections),strata=[])
    for split in cfg['splits']:
        for cls in cfg['event_classes']:
            n=sum(c['split']==split and c['event_class']==cls for c in candidates if not c['rejection']);k=sum(c['split']==split and c['event_class']==cls for c in chosen)
            summary['strata'].append(dict(split=split,event_class=cls,candidates=n,selected=k))
    write_json(OUT/'summary.json',summary);write_json(OUT/'protocol.json',cfg)
    write_json(OUT/'selected.json',dict(version=cfg['version'],source_sha256=source['sha256'],selection_code_sha256=script_sha,config_sha256=config_sha,clips=chosen))
    print(json.dumps(summary,indent=2),flush=True)

def materialize(c,g,lanes,cfg,source,config_sha):
    start=c['start_ms'];end=c['end_ms'];width=cfg['lane_width_m']
    window=g[(g.global_time>=start)&(g.global_time<=end)]
    tracks=[];outside=0
    for ident,tr in window.groupby('vehicle_id'):
        a=tr.sort_values('global_time');valid=a.lane_id.between(1,lanes).to_numpy()
        t=a.global_time.to_numpy();x=(a.local_y-a.v_length/2).to_numpy()*.3048;y=a.local_x.to_numpy()*.3048
        length=a.v_length.to_numpy()*.3048;vw=a.v_width.to_numpy()*.3048
        valid&=np.isfinite(np.c_[x,y,length,vw]).all(axis=1)&(length>0)&(vw>0)&(y>=0)&(y<=lanes*width)
        cuts=np.flatnonzero((np.diff(t)!=100)|(~valid[:-1])|(~valid[1:]))+1
        for k,inds in enumerate(np.split(np.arange(len(a)),cuts)):
            if len(inds)<2 or not valid[inds].all():continue
            ts=t[inds];xx=x[inds];yy=y[inds];vx=np.gradient(xx,.1);vy=np.gradient(yy,.1)
            if vx.min()<cfg.get('longitudinal_velocity_m_s',[-2,55])[0] or vx.max()>cfg.get('longitudinal_velocity_m_s',[-2,55])[1] or abs(vy).max()>cfg.get('lateral_velocity_abs_m_s',8):
                # Do not silently delete corrupted traffic from a selected episode.
                raise ValueError('background_position_jump')
            tracks.append(dict(id=f'{int(ident)}:{k}',source_id=str(int(ident)),ego=int(ident)==c['ego_id'],length=float(np.median(length[inds])),width=float(np.median(vw[inds])),states=np.c_[(ts-start)/1000,xx,(lanes-.5)*width-yy,vx,-vy].tolist()))
    egos=[t for t in tracks if t['ego']]
    if len(egos)!=1 or len(egos[0]['states'])!=201:raise ValueError('ego_incomplete')
    ego=egos[0];ex,ey=ego['states'][0][1:3]
    for tr in tracks:
        if tr['ego'] or tr['states'][0][0]!=0:continue
        x,y=tr['states'][0][1:3]
        if abs(x-ex)<(ego['length']+tr['length'])/2 and abs(y-ey)<(ego['width']+tr['width'])/2:raise ValueError('initial_ego_overlap')
    origin=min(s[1] for tr in tracks for s in tr['states'])-30
    for tr in tracks:
        for s in tr['states']:s[1]-=origin
    extent=max(s[1] for tr in tracks for s in tr['states'])+50
    if extent>900:raise ValueError('road_extent_gt_900m')
    identifier=c['candidate_id'];target=STORE/identifier;target.mkdir(parents=True,exist_ok=True)
    group='ngsim:'+c['recording'];geometry='NGSIM measured mainline positions; equal-width straight approximation; ramps excluded'
    clip=dict(schema_version=1,source='ngsim',ego_id=str(c['ego_id']),duration=20.,lane_num=lanes,lane_width=width,road_length=extent,tracks=tracks,geometry=geometry,recording_group=group,source_start_time=start/1000,source_end_time=end/1000,coordinate_origin_x=origin)
    write_json(target/'clip.json',clip)
    role='train' if c['split']=='train' else 'eval'
    manifest=dict(id=identifier,label=f'NGSIM · {c["split"]} · {c["event_class"]} · {c["location"]} · ego {c["ego_id"]}',source='ngsim',split=role,experiment_split=c['split'],experiment=cfg['version'],event_class=c['event_class'],recording_group=group,source_sha256={'trajectories.csv':source['sha256']},source_path=str(ROOT/'datasets/raw/ngsim/full/trajectories.csv'),clip_sha256=digest(target/'clip.json'),ego_id=str(c['ego_id']),duration=20.,lane_num=lanes,lane_width=width,tracks=len(tracks),geometry=geometry,traffic_mode='recorded_open_loop',source_start_time=start/1000,source_end_time=end/1000,synthetic=False,selection_config_sha256=config_sha,selection_code_sha256=digest(Path(__file__)))
    write_json(target/'manifest.json',manifest)
    return identifier

if __name__=='__main__':main()
