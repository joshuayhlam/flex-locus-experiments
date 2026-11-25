"""Restore missing tail diagnostics and finish roundoff-limited saved solves."""
import json
from pathlib import Path
import numpy as np
from geometry_inputs import make_cases
from target_metric import solve_metric
from harmonic_map import HarmonicMap

HERE=Path(__file__).resolve().parent
def main():
 cases=make_cases()
 for base in ['A','B']:
  source=None;source_level=None
  files=sorted((HERE/'results').glob(base+'_*.json'))
  files.sort(key=lambda p:json.loads(p.read_text())['level'])
  for path in files:
   d=json.loads(path.read_text());name=d['target'];level=d['level']
   if 'tail_corrected_pencil_defect' in d['measurement'] and d['solver']['converged']:continue
   cfg=d['settings'];case=cases[base]
   if source_level!=level:
    source=solve_metric(case.source,**cfg,cache=HERE/'cache'/f'{base}_source_{level}')
    source_level=level
   tar=case.targets[name]
   target=solve_metric(tar.ambient,**dict(cfg,rmin=cfg['rmin']/4),cache=HERE/'cache'/f'{base}_{name}_{level}')
   hm=HarmonicMap(source,target,tar.warp)
   with np.load(path.with_suffix('.npz')) as old:hm.v=old['v'].copy()
   if not d['solver']['converged']:
    d['previous_solver']=d['solver']
    d['solver']=hm.solve(verbose=True,maxiter=15)
    np.savez_compressed(path.with_suffix('.npz'),v=hm.v,d=hm.d,z=hm.M['z'],tri=hm.M['tri'])
   d['measurement']=hm.measure(case.W_coeff)
   path.write_text(json.dumps(d,indent=2)+'\n')
   print(path.name,'finalized',d['solver']['converged'],flush=True)

if __name__=='__main__':main()
