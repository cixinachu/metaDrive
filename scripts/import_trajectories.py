"""Import an authorized highD recording or a public NGSIM highway trajectory file."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from datasets.importer import import_clip

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,choices=['highd','ngsim'])
    p.add_argument('--path',required=True)
    p.add_argument('--ego-id',required=True,type=int)
    p.add_argument('--split',required=True,choices=['train','eval'])
    p.add_argument('--recording-group',default='')
    p.add_argument('--offset',type=float,default=0)
    p.add_argument('--duration',type=float,default=20)
    p.add_argument('--lane-count',type=int,default=5)
    p.add_argument('--lane-width',type=float,default=3.6576)
    print(json.dumps(import_clip(vars(p.parse_args())),ensure_ascii=False,indent=2))
