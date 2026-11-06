import argparse,json,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--config')
a,rest=p.parse_known_args()
defaults=[]
if a.config:
    for k,v in json.loads(Path(a.config).read_text()).items():
        if isinstance(v,bool):
            if v:defaults.append('--'+k)
        elif v is not None:defaults.extend(['--'+k,str(v)])
sys.argv=['blinkmimic.train']+defaults+rest
runpy.run_module('blinkmimic.train',run_name='__main__')
