import argparse
import gc
import json
from pathlib import Path
import numpy as np
from geometry_inputs import make_cases
from target_metric import solve_metric
from harmonic_map import HarmonicMap

HERE=Path(__file__).resolve().parent
LEVELS={
 'coarse':dict(N=16,rmin=1e-3,ntheta=24,ratio=1.35,quad=4),
 'medium':dict(N=24,rmin=1e-3,ntheta=36,ratio=1.25,quad=4),
 'fine':dict(N=36,rmin=1e-3,ntheta=54,ratio=1.18,quad=4),
 'cusp':dict(N=36,rmin=2.5e-4,ntheta=54,ratio=1.18,quad=4),
 'deep':dict(N=36,rmin=6.25e-5,ntheta=54,ratio=1.18,quad=4),
 'finest':dict(N=54,rmin=6.25e-5,ntheta=80,ratio=1.13,quad=4),
 'quadrature':dict(N=54,rmin=6.25e-5,ntheta=80,ratio=1.13,quad=6),
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--bases',nargs='+',default=['A','B'])
 p.add_argument('--targets',nargs='+',default=['flex_1','flex_2','control_1','control_2'])
 p.add_argument('--levels',nargs='+',default=['coarse','medium','fine','cusp'])
 p.add_argument('--verbose',action='store_true');p.add_argument('--check',action='store_true')
 p.add_argument('--maxiter',type=int,default=60)
 p.add_argument('--resume',action='store_true',help='Resume existing saved maps and retain previous solver diagnostics')
 p.add_argument('--initial-level',help='Initialize from a saved map with the same mesh nodes at another quadrature order')
 a=p.parse_args();cases=make_cases()
 (HERE/'results').mkdir(exist_ok=True);(HERE/'cache').mkdir(exist_ok=True)
 for base in a.bases:
  case=cases[base]
  for level in a.levels:
   cfg=LEVELS[level]
   source=solve_metric(case.source,**cfg,cache=HERE/'cache'/f'{base}_source_{level}')
   for name in a.targets:
    target=case.targets[name]
    out=HERE/'results'/f'{base}_{name}_{level}.json'
    if out.exists() and not a.resume: print('EXISTS',out.name,flush=True);continue
    cfg_target=dict(cfg,rmin=cfg['rmin']/4)
    target_metric=solve_metric(target.ambient,**cfg_target,cache=HERE/'cache'/f'{base}_{name}_{level}')
    hm=HarmonicMap(source,target_metric,target.warp)
    result=dict(base=base,target=name,level=level,settings=cfg,source_metric=source.stats,target_metric=target_metric.stats)
    if a.resume and out.exists():
     previous=json.loads(out.read_text())
     result['previous_solver']=previous['solver']
     with np.load(out.with_suffix('.npz')) as saved:hm.v=saved['v'].copy()
    elif a.initial_level:
     prior=HERE/'results'/f'{base}_{name}_{a.initial_level}.npz'
     with np.load(prior) as saved:
      if not np.array_equal(saved['tri'],hm.M['tri']) or np.max(abs(saved['z']-hm.M['z']))>1e-13:
       raise ValueError('Initialization meshes do not match')
      hm.v=saved['v'].copy()
     result['initialized_from_level']=a.initial_level
    if a.check: result['gradient_check']=hm.gradient_check()
    result['solver']=hm.solve(maxiter=a.maxiter,verbose=a.verbose)
    result['measurement']=hm.measure(case.W_coeff)
    out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),v=hm.v,d=hm.d,z=hm.M['z'],tri=hm.M['tri'])
    print(base,name,level,json.dumps(result['measurement']), 'SOLVER',result['solver']['converged'],result['solver']['iterations'],flush=True)
    del hm,target_metric
    gc.collect()

if __name__=='__main__': main()
