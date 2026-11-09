"""Collect native, versioned MetaDrive episodes without overwriting prior runs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.srairl.metadrive_expert import collect_metadrive_expert, save_raw_episode
from datasets.srairl.expert_builder import save_transitions
from env.tasks.srairl_interaction import ROAD_GEOMETRY_CONFIG, SCENARIO_SPEED_CONFIG
from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenarios',nargs='+',choices=tuple(SCENARIO_SPEED_CONFIG),default=tuple(SCENARIO_SPEED_CONFIG))
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--seed',type=int,default=10000)
    parser.add_argument('--vary-initial-state',action='store_true')
    parser.add_argument('--traffic-density',type=float,default=.05)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={}
    for scene in args.scenarios:
        env=SRAIRLProceduralEnv(scene,lane_num=3,traffic_density=args.traffic_density,seed=args.seed,horizon=1000)
        try:
            transitions,rows,raw=collect_metadrive_expert(env,args.episodes,args.seed,vary_initial_state=args.vary_initial_state)
            for episode in raw:save_raw_episode(args.output/scene/f"seed{episode['seed']}.npz",episode)
            if transitions:
                save_transitions(args.output/f'{scene}.npz',transitions,
                    dict(source='MetaDrive',scene_source='procedural',scenario=scene,road_geometry=ROAD_GEOMETRY_CONFIG,
                         speed_config=SCENARIO_SPEED_CONFIG[scene],traffic_density=args.traffic_density),raw)
            report[scene]=dict(episodes=rows,accepted_episodes=sum(r['accepted'] for r in rows),transition_pairs=len(transitions)//2)
            (args.output/'collection.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({scene:report[scene]}),flush=True)
        finally:env.close()


if __name__=='__main__':main()
