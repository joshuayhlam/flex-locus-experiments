"""Numerical transport attempt. Acceptance depends on independent audits.

The connection is computed once per base, seed and refinement stage. It is
then held fixed; only the marked source complex structure is changed. Its
nonzero discrete flatness errors are recorded, not silently identified with
an exact representation.
"""
import argparse,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
import flex_wp as fw
from seed_higgs import Seed,cj
from connection_check import edge_data,return_solve
from spectral_measure import measure
from geometry_inputs import make_cases,ReferenceWarp
from target_metric import solve_metric
from holonomy_audit import audit

ROOT=Path(__file__).parent
LEVELS={'M1':dict(N=24,R=.001,ntheta=36,ratio=1.25),
        'M2':dict(N=48,R=.001,ntheta=64,ratio=1.18),
        'C1':dict(N=48,R=.00025,ntheta=64,ratio=1.18),
        'M3':dict(N=72,R=.00025,ntheta=96,ratio=1.14)}

def make_geometry(case,o,target):
    if target=='source':return o.F,o.M['tz'],None
    tc=case.targets[target];F=fw.Flex(*tc.u);T=F.T
    perm=min(itertools.permutations(range(3)),key=lambda pp:sum(abs(T.center(F.p[j]-r))**2 for j,r in zip(pp,case.chart.ref)))
    origin=case.chart.ref[0]+T.center(F.p[perm[0]]-case.chart.ref[0])
    warp=ReferenceWarp(case.x,tc.x,inner_fraction=.12,outer_fraction=.46)
    def apply(z):return warp(z-o.F.p[0])+origin
    return F,apply(o.M['tz']),apply

def run(base,control,level,targets,amplitude=.04):
    cfg=LEVELS[level];name=f'{base}_{"control" if control else "fibre"}_{level}'
    out=ROOT/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    case=make_cases(include_identity=False)[base]
    o=Seed(base,amplitude=amplitude,control=control,coeff_path=ROOT/'inputs'/f'{base}_coefficients.json',**cfg)
    seed=o.solve();(out/'seed.json').write_text(json.dumps(seed,indent=2)+'\n')
    if not seed['success']:raise RuntimeError('Seed nonlinear solve failed')
    data=edge_data(o,steps=8)
    connhash=hashlib.sha256(data['links'].tobytes()).hexdigest()
    flat=dict(curvature_proxy=data['curvature_proxy'],face_max=float(max(data['triangle_holonomy'])),macroscopic=audit(data,o.F),sha256=connhash)
    (out/'connection_audit.json').write_text(json.dumps(flat,indent=2)+'\n')
    np.savez_compressed(out/'fixed_connection.npz',links=data['links'],edges=data['edges'],tri=data['tri'],tz=data['tz'],z=data['z'],s=o.s,u=o.u)
    first=None;p=None;results=[]
    for target in ['source']+targets+['return']:
        label='source' if target=='return' else target
        F,_,warp=make_geometry(case,o,label)
        tz=data['tz'] if warp is None else warp(data['tz'])
        metric=solve_metric(F,N=cfg['N'],rmin=cfg['R']/4,ntheta=cfg['ntheta'],ratio=cfg['ratio'],quad=4,cache=ROOT/'metric_cache'/f'{name}_{label}')
        p,solver=return_solve(data,start=p,tz=tz)
        measurement=measure(data,p,tz,F,metric)
        record=dict(target=target,base=base,control=control,level=level,solver=solver,measurement=measurement,
                    fixed_connection_sha256=connhash,status='unvalidated_transport_attempt')
        if target=='source':first=p.copy()
        if target=='return':
            c=np.sqrt(1+np.sum(p*p,axis=1))*np.sqrt(1+np.sum(first*first,axis=1))-np.sum(p*first,axis=1)
            record['return_to_first_solve_max']=float(np.arccosh(np.maximum(c,1)).max())
        (out/f'{target}.json').write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(out/f'{target}.npz',p=p)
        results.append(record)
        print(name,target,'eps',measurement['pencil_defect'],'fibre',measurement['normalized_fibre_defect'],'eta',measurement['holomorphic_fit_residual'],'solver',solver['gradient_inf'],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(seed=seed,connection=flat,results=results),indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',choices=['A','B'],default='A');ap.add_argument('--control',action='store_true')
    ap.add_argument('--level',choices=list(LEVELS),default='M1');ap.add_argument('--targets',nargs='*',default=['flex_1','flex_2','flex_b1','flex_b2'])
    a=ap.parse_args();run(a.base,a.control,a.level,a.targets)
