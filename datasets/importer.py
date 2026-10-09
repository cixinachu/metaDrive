"""Convert highway trajectory windows to SI vehicle-center coordinates.

Roads are straight approximations. Background trajectories are open-loop replay.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT/'datasets/processed'
NGSIM_COLUMNS = 'vehicle_id frame_id total_frames global_time local_x local_y global_x global_y v_length v_width v_class v_vel v_acc lane_id preceding following space_headway time_headway'.split()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        first = f.readline(); f.seek(0)
        if ',' in first:
            for row in csv.DictReader(f):
                yield {k.strip().lower(): v.strip() for k,v in row.items() if k is not None and v is not None}
        else:
            for line in f:
                fields = line.split()
                if not fields: continue
                if len(fields) != 18:
                    raise ValueError('NGSIM 无表头文本必须为标准 18 列')
                yield dict(zip(NGSIM_COLUMNS, fields))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def catalog(store=STORE):
    result=[]
    for p in sorted(Path(store).glob('*/manifest.json')):
        result.append(json.loads(p.read_text(encoding="utf-8")))
    return result


def load_clip(identifier, store=STORE):
    if not isinstance(identifier,str) or len(identifier)!=24 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('请选择已导入的数据片段')
    p=Path(store)/identifier
    manifest=json.loads((p/'manifest.json').read_text(encoding="utf-8"))
    if digest(p/'clip.json') != manifest['clip_sha256']: raise ValueError('数据片段校验失败')
    return json.loads((p/'clip.json').read_text(encoding="utf-8")),manifest


def import_clip(options, store=STORE):
    source=options['source']
    if source not in ('highd','ngsim'): raise ValueError('数据源必须为 highd 或 ngsim')
    split=options.get('split','eval')
    if split not in ('train','eval'): raise ValueError('split 必须为 train 或 eval')
    path=Path(options['path']).expanduser().resolve()
    if not path.is_file(): raise ValueError('轨迹文件不存在')
    ego=str(int(options['ego_id']))
    offset=float(options.get('offset',0)); duration=float(options.get('duration',20))
    if not math.isfinite(offset) or offset<0 or not math.isfinite(duration) or not 1<=duration<=120:
        raise ValueError('起始偏移必须非负，片段长度须为 1–120 秒')
    hashes={path.name:digest(path)}
    direction=1
    if source=='highd':
        if not path.name.endswith('_tracks.csv'): raise ValueError('HighD 请选择 XX_tracks.csv，元数据须位于同目录')
        prefix=path.name[:-len('_tracks.csv')]
        meta_path=path.with_name(prefix+'_recordingMeta.csv'); tracks_path=path.with_name(prefix+'_tracksMeta.csv')
        rec=next(rows(meta_path)); meta={str(int(r['id'])):r for r in rows(tracks_path)}
        for p in (meta_path,tracks_path): hashes[p.name]=digest(p)
        if ego not in meta: raise ValueError('找不到自车轨迹 ID')
        fps=float(rec['framerate']); direction=1 if int(meta[ego]['drivingdirection'])==2 else -1
        boundaries=sorted(float(x) for x in rec['lowerlanemarkings' if direction==1 else 'upperlanemarkings'].split(';'))
        widths=np.diff(boundaries); lane_count=len(widths); lane_width=float(np.mean(widths))
        if fps<=0 or not np.isfinite(widths).all() or np.any(widths<=0): raise ValueError('HighD 帧率或车道边界无效')
        if np.max(np.abs(widths-lane_width))>.3: raise ValueError('当前直道重建仅支持近似等宽车道（差异 ≤ 0.3m）')
        def decode(r):
            ident=str(int(r['id'])); length=float(r['width']); width=float(r['height'])
            cy=float(r['y'])+width/2
            # Reflect image coordinates into forward/right lane coordinates.
            lateral=cy-boundaries[0] if direction==1 else boundaries[-1]-cy
            return ident,float(r['frame'])/fps,direction*(float(r['x'])+length/2),lateral,length,width
        def eligible(r):
            return str(int(r['id'])) in meta and (int(meta[str(int(r['id']))]['drivingdirection'])==2)==(direction==1)
        recording_group=f'highd:{rec["id"]}'
        geometry='HighD lane markings; equal-width straight-road approximation'
    else:
        lane_count=int(options.get('lane_count',5)); lane_width=float(options.get('lane_width',3.6576)); fps=10.
        road=options.get('road_type','highway')
        if road!='highway':
            raise ValueError('此导入器仅支持已验证的高速公路直路片段；城市交叉口和匝道场景需先完成道路网络重建')
        lane_count=int(options.get('lane_count',5)); lane_width=float(options.get('lane_width',3.6576)); fps=10.
        def decode(r):
            length=float(r['v_length'])*.3048; width=float(r['v_width'])*.3048
            return str(int(r['vehicle_id'])),float(r['global_time'])/1000,(float(r['local_y'])*.3048-length/2),float(r['local_x'])*.3048,length,width
        def eligible(r):
            if r.get('location','').lower() in ('peachtree','lankershim'):
                raise ValueError('当前导入器不支持城市交叉口数据；请使用已验证的道路网络重建版本')
            return 1<=int(r['lane_id'])<=lane_count
        recording_group=options.get('recording_group','')
        if not recording_group: raise ValueError('NGSIM 必须指定采集组（如 us101-20050615），防止同一采集记录跨训练测试集')
        recording_group='ngsim:'+recording_group
        geometry='NGSIM Local_X/Local_Y; user-specified straight mainline'
    if not 1<=lane_count<=8 or not math.isfinite(lane_width) or not 2<=lane_width<=6: raise ValueError('数据道路要求 1–8 车道，宽度 2–6m')
    # Do not let two overlapping excerpts of a recording leak across the split.
    for m in catalog(store):
        if (m['recording_group']==recording_group or hashes[path.name] in m['source_sha256'].values()) and m['split']!=split:
            raise ValueError('同一采集组/原始文件已用于另一数据划分，请使用独立采集记录')
    ego_rows=[]
    for r in rows(path):
        id_key='id' if source=='highd' else 'vehicle_id'
        if str(int(r[id_key]))==ego and eligible(r): ego_rows.append(decode(r))
    if len(ego_rows)<2: raise ValueError('该自车不存在或不在所选主线车道内')
    ego_rows.sort(key=lambda r:r[1]);start=ego_rows[0][1]+offset;end=min(start+duration,ego_rows[-1][1])
    if end-start<1: raise ValueError('所选自车剩余连续轨迹不足 1 秒')
    tracks={}
    for r in rows(path):
        if not eligible(r): continue
        ident,t,x,y,length,width=decode(r)
        if not start-1/fps<=t<=end+1/fps: continue
        if not all(math.isfinite(v) for v in (t,x,y,length,width)) or length<=0 or width<=0: raise ValueError('发现非有限数值或无效车辆尺寸')
        tracks.setdefault(ident,[]).append([t,x,y,length,width])
    segments=[]
    duplicates_removed=0
    for ident,rr in tracks.items():
        unique=sorted(set(tuple(r) for r in rr),key=lambda r:r[0])
        duplicates_removed+=len(rr)-len(unique)
        a=np.asarray(unique)
        if np.any(np.diff(a[:,0])<=0): raise ValueError(f'轨迹 {ident} 存在重复时间；请拆分不同采集时段/地点')
        groups=np.split(a,np.where(np.diff(a[:,0])>1.6/fps)[0]+1)
        for part,arr in enumerate(groups):
            if len(arr)<2: continue
            if ident==ego and not (arr[0,0]<=start+1e-6 and arr[-1,0]>=end-1e-6): continue
            segments.append((ident,part,arr))
    ego_seg=next((a for i,p,a in segments if i==ego),None)
    if ego_seg is None: raise ValueError('自车轨迹有缺帧或跨时段，缩短片段或调整偏移')
    x_origin=min(float(a[:,1].min()) for _,_,a in segments)-30
    normalized=[]
    for ident,part,a in segments:
        # Keep measured positions; velocities are derivatives, no implicit smoothing.
        vx=np.gradient(a[:,1],a[:,0]); vy=np.gradient(a[:,2],a[:,0])
        states=np.c_[a[:,0]-start,a[:,1]-x_origin,(lane_count-.5)*lane_width-a[:,2],vx,-vy]
        normalized.append(dict(id=f'{ident}:{part}',source_id=ident,ego=ident==ego,length=float(np.median(a[:,3])),width=float(np.median(a[:,4])),states=states.tolist()))
    extent=max(max(s[1] for s in t['states']) for t in normalized)+50
    if extent>900: raise ValueError('当前地图显示范围支持最长 900m 的道路片段，请先裁剪输入轨迹范围')
    clip=dict(schema_version=1,source=source,ego_id=ego,duration=end-start,lane_num=lane_count,lane_width=lane_width,
              road_length=extent,tracks=normalized,geometry=geometry,recording_group=recording_group,
              source_start_time=start,source_end_time=end,coordinate_origin_x=x_origin)
    identity=dict(source=source,source_sha256=hashes,ego_id=ego,start=start,end=end,split=split,lanes=lane_count,lane_width=lane_width)
    identifier=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    target=Path(store)/identifier;target.mkdir(parents=True,exist_ok=True)
    content=json.dumps(clip,allow_nan=False);(target/'clip.json').write_text(content, encoding="utf-8")
    manifest=dict(id=identifier,label=f'{source.upper()} · {recording_group} · ego {ego} · {end-start:.1f}s · {split}',
                  source=source,split=split,recording_group=recording_group,source_sha256=hashes,source_path=str(path),
                  clip_sha256=digest(target/'clip.json'),ego_id=ego,duration=end-start,lane_num=lane_count,lane_width=lane_width,
                  tracks=len(normalized),duplicates_removed=duplicates_removed,geometry=geometry,traffic_mode='recorded_open_loop',source_start_time=start,source_end_time=end,
                  synthetic=bool(options.get('synthetic',False)))
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False), encoding="utf-8")
    return manifest
