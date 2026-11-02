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
CONTROLS=np.unique(np.round(np.concatenate((np.arange(-1.,1.001,.1),np.arange(-.04,.0401,.002))),3))
DESIRED_ACCELS=(-1.,0.,1.)
DT=.1
WINDOW=3


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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('outputs/calibration/srairl_action_adapter_v2'))
    p.add_argument('--seed',type=int,default=2718)
    p.add_argument('--mae-limit',type=float,default=.15)
    p.add_argument('--max-error-limit',type=float,default=.30)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
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
    metrics={};passed=True;inverse_rows=[]
    for axis,values in surfaces.items():
        fig,ax=plt.subplots(figsize=(9,5))
        for i,speed in enumerate(SPEEDS):
            ax.plot(CONTROLS,values[i],label=f'{speed:g} m/s')
            for desired in DESIRED_ACCELS:
                control=_inverse_surface(CONTROLS,values[i],desired)
                actual=measure(env,float(speed),control,axis)
                inverse_rows.append({'axis':axis,'speed_mps':float(speed),'desired_accel_mps2':desired,
                    'control':control,'actual_accel_mps2':actual,'error_mps2':actual-desired,
                    'absolute_error_mps2':abs(actual-desired)})
        ax.axhline(0,color='#666',lw=.8);ax.set(xlabel='MetaDrive control',ylabel='Measured acceleration (m/s²)',title=f'{axis.title()} v-control response')
        ax.grid(alpha=.25);ax.legend(ncol=3);fig.tight_layout();fig.savefig(figures/f'{axis}.png',dpi=160);plt.close(fig)
        selected=[r for r in inverse_rows if r['axis']==axis]
        errors=[r['absolute_error_mps2'] for r in selected];max_error=max(errors);mae=float(np.mean(errors))
        wrong_sign=sum(abs(r['desired_accel_mps2'])>=.5 and np.sign(r['actual_accel_mps2'])!=np.sign(r['desired_accel_mps2']) for r in selected)
        axis_passed=mae<args.mae_limit and max_error<args.max_error_limit and wrong_sign==0
        metrics[axis]={'inverse_validation_mae_mps2':mae,'inverse_validation_max_error_mps2':max_error,'wrong_sign_count':int(wrong_sign),'passed':bool(axis_passed)}
        passed=bool(passed and axis_passed)
    with (out/'inverse_validation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(inverse_rows[0]));writer.writeheader();writer.writerows(inverse_rows)
    env.close()
    try:version=importlib.metadata.version('metadrive-simulator')
    except importlib.metadata.PackageNotFoundError:version='unknown'
    summary={'simulator':'MetaDrive','simulator_version':version,'seed':args.seed,'speeds_mps':SPEEDS.tolist(),
        'controls':CONTROLS.tolist(),'dt_s':DT,'measurement_window_steps':WINDOW,
        'vehicle_model':'MetaDrive default vehicle','max_speed_km_h':130,'natural_coast_included':True,
        'acceptance':{'mae_mps2_lt':args.mae_limit,'max_error_mps2_lt':args.max_error_limit,'sign_correct_for_abs_desired_ge_mps2':.5},
        'metrics':metrics,'equivalence_status':'passed' if passed else 'failed',
        'physics_evaluation_allowed':bool(passed),'calibration_file':'calibration.npz'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
