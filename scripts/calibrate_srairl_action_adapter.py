"""Identify MetaDrive's speed/control-to-acceleration surfaces for SR-AIRL."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
from env.adapters.srairl_action import _inverse_surface

SPEEDS=np.arange(5.,31.,5.)
HELD_OUT_SPEEDS=np.arange(7.5,30.,5.)
CONTROLS=np.unique(np.round(np.concatenate((np.arange(-1.,1.0001,.1),
                                            np.arange(-.04,.0401,.002))),3))
DESIRED_ACCELS=(-1.5,-1.,-.5,0.,.5,1.,1.5)
DT=.1
WINDOW=5


def measure(env,speed,control,axis):
    env.reset()
    vehicle=env.world.role_vehicle('elcv')
    lane=vehicle.lane
    longitudinal,_=lane.local_coordinates(vehicle.position[:2])
    heading=float(lane.heading_theta_at(longitudinal))
    tangent=np.asarray([np.cos(heading),np.sin(heading)])
    normal=np.asarray([-tangent[1],tangent[0]])
    vehicle.set_heading_theta(heading)
    vehicle.set_velocity(tangent*speed)
    initial=np.asarray(vehicle.velocity[:2],dtype=float)
    # MetaDrive action order is [steering, throttle/brake].
    md_action=[0.,control] if axis=='longitudinal' else [control,0.]
    for _ in range(WINDOW):
        env.world.step({'elcv':md_action,'rfv':[0.,0.]})
    final=np.asarray(vehicle.velocity[:2],dtype=float)
    basis=tangent if axis=='longitudinal' else normal
    return float((final-initial)@basis/(WINDOW*DT))


def dump_csv(path,axis,values):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['speed_mps','control','actual_accel_mps2','coast_accel_mps2'])
        for i,speed in enumerate(SPEEDS):
            coast=float(values[i,np.where(CONTROLS==0.)[0][0]])
            for j,control in enumerate(CONTROLS):writer.writerow([speed,control,values[i,j],coast])


def inverse_at_speed(speed,desired,speeds,controls,response):
    """Interpolate the measured surface in speed, then invert its control axis."""
    hi=int(np.clip(np.searchsorted(speeds,speed),1,len(speeds)-1));lo=hi-1
    weight=float(np.clip((speed-speeds[lo])/(speeds[hi]-speeds[lo]),0.,1.))
    row=(1.-weight)*response[lo]+weight*response[hi]
    return _inverse_surface(controls,row,float(desired))


def validate_surface(env,axis,speeds,controls,response,desired_accels):
    rows=[]
    for speed in speeds:
        for desired in desired_accels:
            control=inverse_at_speed(float(speed),float(desired),SPEEDS,controls,response)
            actual=measure(env,float(speed),control,axis)
            rows.append({'axis':axis,'speed_mps':float(speed),'desired_accel_mps2':float(desired),
                         'control':control,'actual_accel_mps2':actual,'error_mps2':actual-float(desired),
                         'absolute_error_mps2':abs(actual-float(desired))})
    errors=[row['absolute_error_mps2'] for row in rows]
    wrong_sign=sum(abs(row['desired_accel_mps2'])>=.5 and
                   np.sign(row['actual_accel_mps2'])!=np.sign(row['desired_accel_mps2']) for row in rows)
    metrics={'mae_mps2':float(np.mean(errors)),'max_error_mps2':float(max(errors)),
             'wrong_sign_count':int(wrong_sign)}
    return rows,metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('outputs/calibration/srairl_action_adapter_v3'))
    p.add_argument('--seed',type=int,default=2718)
    p.add_argument('--mae-limit',type=float,default=.15)
    p.add_argument('--max-error-limit',type=float,default=.30)
    p.add_argument('--closed-loop-report',type=Path)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    env=SRAIRLProceduralEnv('straight',lane_num=2,traffic_density=0.,seed=args.seed,
                            horizon=WINDOW+2,traffic_seed=args.seed+1000003)
    surfaces={}
    for axis in ('longitudinal','lateral'):
        values=np.zeros((len(SPEEDS),len(CONTROLS)),dtype=np.float64)
        for i,speed in enumerate(SPEEDS):
            for j,control in enumerate(CONTROLS):values[i,j]=measure(env,float(speed),float(control),axis)
        surfaces[axis]=values
    dump_csv(out/'longitudinal.csv','longitudinal',surfaces['longitudinal'])
    dump_csv(out/'lateral.csv','lateral',surfaces['lateral'])
    np.savez_compressed(out/'calibration.npz',speeds=SPEEDS,controls=CONTROLS,
        longitudinal_accel=surfaces['longitudinal'],lateral_accel=surfaces['lateral'])
    figures=out/'figures';figures.mkdir(exist_ok=True)
    metrics={};passed=True;grid_rows=[];held_out_rows=[]
    for axis,values in surfaces.items():
        fig,ax=plt.subplots(figsize=(9,5))
        for i,speed in enumerate(SPEEDS):
            ax.plot(CONTROLS,values[i],label=f'{speed:g} m/s')
        ax.axhline(0,color='#666',lw=.8);ax.set(xlabel='MetaDrive control',ylabel='Measured acceleration (m/s²)',title=f'{axis.title()} v-control response')
        ax.grid(alpha=.25);ax.legend(ncol=3);fig.tight_layout();fig.savefig(figures/f'{axis}.png',dpi=160);plt.close(fig)
        grid,grid_metrics=validate_surface(env,axis,SPEEDS,CONTROLS,values,DESIRED_ACCELS)
        held,held_metrics=validate_surface(env,axis,HELD_OUT_SPEEDS,CONTROLS,values,DESIRED_ACCELS)
        grid_rows.extend(grid);held_out_rows.extend(held)
        for result in (grid_metrics,held_metrics):
            result['passed']=bool(result['mae_mps2']<args.mae_limit and
                                  result['max_error_mps2']<args.max_error_limit and
                                  result['wrong_sign_count']==0)
        axis_passed=grid_metrics['passed'] and held_metrics['passed']
        metrics[axis]={'calibration_grid':grid_metrics,'held_out':held_metrics,'passed':bool(axis_passed)}
        passed=bool(passed and axis_passed)
    for filename,rows in (('inverse_validation.csv',grid_rows),('held_out_validation.csv',held_out_rows)):
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    env.close()
    try:version=importlib.metadata.version('metadrive-simulator')
    except importlib.metadata.PackageNotFoundError:version='unknown'
    summary={'simulator':'MetaDrive','simulator_version':version,'seed':args.seed,'speeds_mps':SPEEDS.tolist(),
        'held_out_speeds_mps':HELD_OUT_SPEEDS.tolist(),'desired_accelerations_mps2':list(DESIRED_ACCELS),
        'controls':CONTROLS.tolist(),'dt_s':DT,'measurement_window_steps':WINDOW,
        'vehicle_model':'MetaDrive default vehicle','max_speed_km_h':130,'natural_coast_included':True,
        'acceptance':{'mae_mps2_lt':args.mae_limit,'max_error_mps2_lt':args.max_error_limit,'sign_correct_for_abs_desired_ge_mps2':.5},
        'validated_acceleration_domain':{'ax_mps2':[-1.5,1.5], 'ay_mps2':[-1.5,1.5]},
        'metrics':metrics,'equivalence_status':'passed' if passed else 'failed',
        'physics_evaluation_allowed':bool(passed),'calibration_file':'calibration.npz'}
    from env.adapters.srairl_calibration import action_calibration_passed
    if args.closed_loop_report:
        summary['closed_loop_report']=json.loads(args.closed_loop_report.read_text())
    summary['physics_evaluation_allowed']=action_calibration_passed(summary,require_closed_loop=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
