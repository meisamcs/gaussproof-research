import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussproof.diffusion_lab import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.source,a.data,a.output,resume=a.resume)
