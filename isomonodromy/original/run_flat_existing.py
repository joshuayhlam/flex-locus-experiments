"""Transport using a genuinely flat cellular bundle, including source audit."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import flex_wp as fw
from seed_higgs import BASES
from flat_bundle import reconstruct
from connection_check import return_solve
from spectral_measure import measure
from run_transport import LEVELS,make_geometry
from geometry_inputs import make_cases
from target_metric import solve_metric

ROOT=Path(__file__).parent

def free_nodes(data):
    counts={tuple(e):0 for e in data['edges']}
    for row in data['tri']:
        for a,b in [(0,1),(1,2),(2,0)]:counts[tuple(sorted((int(row[a]),int(row[b]))))]+=1
    boundary=np.unique(np.array([e for e,n in counts.items() if n==1]))
    return np.setdiff1d(np.arange(len(data['z'])),boundary)

def run(base,control,level):
    name=f'{base}_{"control" if control else "fibre"}_{level}';old=ROOT/'runs'/name
    out=ROOT/'flat_runs'/name;out.mkdir(parents=True,exist_ok=True)
    saved=np.load(old/'fixed_connection.npz');data={k:saved[k] for k in ['edges','links','tri','tz','z']}
    F=fw.Flex(*BASES[base]);U=saved['u'].copy()
    for q in F.p:U-=F.T.green(data['z']-q)[0]
    data['node_U']=U;data['free']=free_nodes(data)
    data['links'],audit,extra=reconstruct(data,F)
    (out/'flat_projection.json').write_text(json.dumps(audit,indent=2)+'\n')
    np.savez_compressed(out/'flat_connection.npz',**{k:data[k] for k in ['edges','links','tri','tz','z','free']})
    sha=hashlib.sha256(data['links'].tobytes()).hexdigest()
    case=make_cases(include_identity=False)[base];o=SimpleNamespace(F=F,M=dict(tz=data['tz']));cfg=LEVELS[level]
    p=np.load(old/'source.npz')['p'];first=None;records=[]
    for target in ['source','flex_1','flex_2','flex_b1','flex_b2','return']:
        label='source' if target=='return' else target
        FF,tz,warp=make_geometry(case,o,label)
        metric=solve_metric(FF,N=cfg['N'],rmin=cfg['R']/4,ntheta=cfg['ntheta'],ratio=cfg['ratio'],quad=4,cache=ROOT/'metric_cache'/f'{name}_{label}')
        p,solver=return_solve(data,start=p,tz=tz)
        m=measure(data,p,tz,FF,metric)
        rec=dict(base=base,control=control,level=level,target=target,solver=solver,measurement=m,connection_sha256=sha,
                 status='exact_flat_bundle_source_membership_requires_audit')
        if target=='source':first=p.copy()
        if target=='return':
            rec['return_coordinate_max']=float(np.max(np.linalg.norm(p-first,axis=1)))
            rec['return_pencil_difference']=abs(m['pencil_defect']-records[0]['measurement']['pencil_defect'])
            rec['return_fibre_difference']=abs(m['normalized_fibre_defect']-records[0]['measurement']['normalized_fibre_defect'])
        np.savez_compressed(out/f'{target}.npz',p=p)
        (out/f'{target}.json').write_text(json.dumps(rec,indent=2)+'\n');records.append(rec)
        print(name,target,'pencil',m['pencil_defect'],'fibre',m['normalized_fibre_defect'],'eta',m['holomorphic_fit_residual'],'source_metric_distance',solver['return_point_distance_max'],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(flatness=audit,results=records),indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='A');ap.add_argument('--control',action='store_true');ap.add_argument('--levels',nargs='+',default=['M1','M2','C1']);a=ap.parse_args()
    for lev in a.levels:run(a.base,a.control,lev)
