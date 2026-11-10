"""Plot existing NGSIM training clips without modifying data or running simulations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter


def load_training_clips(selections):
    clips=[];seen=set()
    for selection in selections:
        rows=json.loads(selection.read_text())['clips']
        for row in rows:
            ident=row.get('clip_id')
            if row.get('split')!='train' or not ident or ident in seen:continue
            folder=ROOT/'datasets/processed'/ident
            manifest=json.loads((folder/'manifest.json').read_text())
            if manifest.get('source')!='ngsim' or manifest.get('split')!='train' or manifest.get('synthetic'):
                raise ValueError(f'Not a real NGSIM training clip: {ident}')
            content=(folder/'clip.json').read_bytes()
            if hashlib.sha256(content).hexdigest()!=manifest['clip_sha256']:
                raise ValueError(f'Clip fingerprint mismatch: {ident}')
            clip=json.loads(content)
            seen.add(ident);clips.append((row,clip,manifest))
    if not clips:raise ValueError('No NGSIM training clips found')
    return clips


def ego_states(clip):
    tracks=[track for track in clip['tracks'] if track.get('ego')]
    if len(tracks)!=1:raise ValueError('Expected exactly one ego trajectory')
    return tracks[0],np.asarray(tracks[0]['states'],dtype=float)


def make_plots(clips,output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white','axes.facecolor':'white'})
    example=next(item for item in clips if item[0]['event_class']=='ego_lane_change')
    row,clip,manifest=example
    ego,states=ego_states(clip)
    t,x,y,vx,vy=states.T
    width=float(clip['lane_width'])
    collected={}
    for _,current,meta in clips:
        track,values=ego_states(current)
        for state in values:
            # Windows may overlap; count a source vehicle at an absolute time only once.
            key=(meta['recording_group'],track.get('source_id',track['id']),
                int(round((float(current['source_start_time'])+state[0])*10)))
            collected.setdefault(key,float(state[3]))
    speeds=np.asarray(list(collected.values()))
    if not np.isfinite(speeds).all():raise ValueError('Nonfinite speed samples')
    figures=[]
    def figure(name,title):
        fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained')
        ax.set_title(title,pad=14)
        figures.append((name,fig));return fig,ax
    def lane_lines(ax,horizontal=True):
        for index in range(int(clip['lane_num'])+1):
            value=(index-.5)*width
            (ax.axhline if horizontal else ax.axvline)(value,color='#cbd5e1',lw=.8,ls='--')
    fig,ax=figure('01_spatial_trajectories','Measured trajectories in one NGSIM lane-change clip')
    lane_lines(ax)
    for track in clip['tracks']:
        s=np.asarray(track['states'])
        if track.get('ego'):continue
        ax.plot(s[:,1],s[:,2],color='#94a3b8',alpha=.35,lw=.8)
    ax.plot(x,y,color='#dc2626',lw=2.5,label=f'Ego source vehicle {ego.get("source_id",ego["id"])}')
    ax.scatter(x[0],y[0],s=45,color='#16a34a',zorder=4,label='Ego start')
    ax.set(xlabel='Longitudinal position in clip coordinates (m)',ylabel='Lateral position (m)')
    ax.legend(loc='upper right');fig.supxlabel('Axes use different visual scales; equal-width straight-road approximation.',fontsize=9)
    fig,ax=figure('02_time_space','Time-space diagram: vehicle motion and traffic interactions')
    for track in clip['tracks']:
        s=np.asarray(track['states'])
        if not track.get('ego'):ax.plot(s[:,0],s[:,1],color='#94a3b8',alpha=.4,lw=.8)
    ax.plot(t,x,color='#dc2626',lw=2.5,label='Ego')
    ax.set(xlabel='Time within clip (s)',ylabel='Longitudinal position (m)');ax.legend()
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    figures.append(('03_speed_acceleration',fig))
    smooth=savgol_filter(vx,11,2)
    acceleration=np.gradient(smooth,t)
    axes[0].plot(t,vx,color='#94a3b8',label='Imported longitudinal velocity')
    axes[0].plot(t,smooth,color='#2563eb',lw=2,label='Savitzky-Golay: 11 samples, degree 2')
    axes[0].set(ylabel='Longitudinal speed (m/s)',title='Ego speed and derived longitudinal acceleration');axes[0].legend()
    axes[1].plot(t,acceleration,color='#ea580c');axes[1].axhline(0,color='#64748b',lw=.8)
    axes[1].set(xlabel='Time within clip (s)',ylabel='Derived acceleration (m/s²)')
    fig.supxlabel('Acceleration is differentiated from smoothed imported velocity; diagnostic, not a physical-action label.',fontsize=9)
    fig,ax=figure('04_lane_change','Lane-change process: continuous lateral position')
    lane_lines(ax)
    for index in range(int(clip['lane_num'])):ax.axhline(index*width,color='#e2e8f0',lw=.7)
    ax.plot(t,y,color='#7c3aed',lw=2.5)
    ax.set(xlabel='Time within clip (s)',ylabel='Lateral position (m)',
        ylim=(float(y.min())-.75*width,float(y.max())+.75*width))
    fig.supxlabel('Geometric lane centers/boundaries are inferred from the imported equal-width map, not original Lane_ID.',fontsize=9)
    fig,ax=figure('05_training_speed_distribution','Ego longitudinal speed distribution in selected training clips')
    ax.hist(speeds,bins=45,density=True,color='#60a5fa',edgecolor='white',alpha=.9)
    percentiles=np.percentile(speeds,[10,50,90])
    for q,value,color in zip((10,50,90),percentiles,('#16a34a','#dc2626','#7c3aed')):
        ax.axvline(value,color=color,lw=1.7,label=f'P{q}: {value:.2f} m/s')
    ax.set(xlabel='Longitudinal speed (m/s)',ylabel='Probability density');ax.legend()
    fig.supxlabel(f'{len(clips)} event-selected training clips; {len(speeds):,} unique ego vehicle-time samples. Not full NGSIM traffic.',fontsize=9)
    fig,ax=figure('06_target_lane_gaps','Nearest target-lane longitudinal center gaps: geometric diagnostic')
    # This is a framewise neighborhood diagnostic, not validated fixed-role extraction.
    target=int(round(float(np.median(y[-20:]))/width))
    front=[];rear=[]
    others=[track for track in clip['tracks'] if not track.get('ego')]
    for time,ego_x in zip(t,x):
        offsets=[]
        for track in others:
            s=np.asarray(track['states'])
            if time<s[0,0] or time>s[-1,0]:continue
            other_y=np.interp(time,s[:,0],s[:,2])
            if int(round(other_y/width))!=target:continue
            offsets.append(float(np.interp(time,s[:,0],s[:,1])-ego_x))
        front.append(min((d for d in offsets if d>0),default=np.nan))
        rear.append(min((-d for d in offsets if d<0),default=np.nan))
    ax.plot(t,front,label='Nearest vehicle ahead',color='#2563eb',lw=2)
    ax.plot(t,rear,label='Nearest vehicle behind',color='#ea580c',lw=2)
    ax.set(xlabel='Time within clip (s)',ylabel='Longitudinal center-to-center gap (m)');ax.legend()
    fig.supxlabel('Nearest identities may change. Gaps are not bumper clearance, fixed RFV/RLV roles, TTC, or safety labels.',fontsize=9)
    with PdfPages(output/'ngsim_analysis.pdf') as pdf:
        for name,fig in figures:
            fig.savefig(output/(name+'.png'),dpi=180)
            fig.savefig(output/(name+'.svg'))
            pdf.savefig(fig);plt.close(fig)
    with (output/'selected_training_clips.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['clip_id','recording_group','event_class','clip_sha256'])
        writer.writeheader()
        for selection,_,meta in clips:
            writer.writerow(dict(clip_id=meta['id'],recording_group=meta['recording_group'],
                event_class=selection['event_class'],clip_sha256=meta['clip_sha256']))
    summary=dict(training_clips=len(clips),unique_ego_vehicle_time_samples=len(speeds),
        speed_mps_percentiles={f'p{q}':float(value) for q,value in zip((10,50,90),percentiles)},
        example_clip_id=manifest['id'],example_source_ego_id=ego.get('source_id',ego['id']),
        example_recording=manifest['recording_group'],example_class=row['event_class'],
        units='Imported clips already use SI units; no second feet-to-metre conversion applied.',
        limitations=['Event-selected training subset, not an unbiased sample of all NGSIM traffic.',
            'Repeated ego vehicle-time samples across overlapping clips deduplicated by recording/source vehicle/absolute 0.1-second timestamp.',
            'Other vehicles and whole-network density are not represented by the ego speed histogram.',
            'Geometry is an equal-width straight approximation; raw ramps are excluded from these clips.',
            'Gap plot uses framewise nearest identities and longitudinal center gaps, not validated four-role interaction samples.',
            'Derived acceleration is smoothed numerical differentiation and is not a calibrated vehicle action.'])
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selections',type=Path,nargs='+',default=[ROOT/f'datasets/experiments/ngsim-mainline-v{i}/selected.json' for i in (1,2)])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'outputs/ngsim_analysis'/(
        datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    summary=make_plots(load_training_clips(args.selections),output)
    print(json.dumps(dict(output=str(output),**summary),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
