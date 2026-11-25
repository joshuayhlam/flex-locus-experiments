"""One exactly flat representation per seed, unchanged across all meshes."""
import argparse,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import flex_wp as fw
from seed_higgs import BASES
from run_transport import LEVELS,make_geometry
from run_flat_existing import free_nodes
from flat_interpolate import FlatInterpolator,face_errors
from connection_check import return_solve
from spectral_measure import measure
from geometry_inputs import make_cases
from target_metric import solve_metric
from cusp_boundary import fit_boundary

ROOT=Path(__file__).parent
CONFIG={**LEVELS,'F':dict(N=64,R=.0000625,ntheta=80,ratio=1.18)}

def geometry_data(F,cfg):
    fw.quadrature_order(4);M=fw.mesh(F,cfg['N'],cfg['R'],cfg['ntheta'],cfg['ratio']);fw.quadratic_elements(F,M)
    sub=np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
    tri=M['tri'][:,sub].reshape(-1,3)
    mids=.5*(M['tz']+np.roll(M['tz'],-1,axis=1));allz=np.concatenate((M['tz'],mids),axis=1)
    tz=allz[:,sub].reshape(-1,3);edge={}
    for row in tri:
        for a,b in [(0,1),(1,2),(2,0)]:edge.setdefault(tuple(sorted((int(row[a]),int(row[b])))),None)
    return dict(edges=np.array(list(edge)),tri=tri,tz=tz,z=M['z'],free=np.setdiff1d(np.arange(M['nv']),np.unique(M['be'])))

def run(base,control,levels,cusp=False):
    name=f'{base}_{"control" if control else "fibre"}';canonical=ROOT/'canonical'/f'{name}_N64'
    saved=np.load(canonical/'flat_connection.npz');fine={k:saved[k] for k in saved.files}
    rhohash=hashlib.sha256(fine['generators'].tobytes()).hexdigest()
    F=fw.Flex(*BASES[base]);case=make_cases(include_identity=False)[base];interpolator=None
    coefficients=json.loads((ROOT/'inputs'/f'{base}_coefficients.json').read_text())
    intended=np.array([1/coefficients['norms'][0],(.4+.3j)/coefficients['norms'][1],0j])
    for level in levels:
        cfg=CONFIG[level];out=ROOT/('cusp_fixed_runs' if cusp else 'fixed_runs')/f'{name}_{level}';out.mkdir(parents=True,exist_ok=True)
        file=out/'connection.npz'
        if file.exists():
            saved=np.load(file);data={k:saved[k] for k in saved.files}
        elif cusp and (ROOT/'fixed_runs'/f'{name}_{level}'/'connection.npz').exists():
            saved=np.load(ROOT/'fixed_runs'/f'{name}_{level}'/'connection.npz');data={k:saved[k] for k in saved.files}
        elif level=='F':
            data={k:fine[k] for k in ['edges','links','tri','tz','z','free']}
        else:
            oldfile=ROOT/'runs'/f'{name}_{level}'/'fixed_connection.npz'
            if oldfile.exists():
                saved=np.load(oldfile);data={k:saved[k] for k in ['edges','tri','tz','z']};data['free']=free_nodes(data)
            else:data=geometry_data(F,cfg)
            if interpolator is None:interpolator=FlatInterpolator(fine,F.T)
            start=time.time();data['links'],walk=interpolator.resample(data['edges'],data['z'])
            walk['seconds']=time.time()-start;(out/'resampling.json').write_text(json.dumps(walk,indent=2)+'\n')
        audit=face_errors(data)
        if audit['maximum']>1e-8:raise RuntimeError(f'Resampling broke flatness: {audit}')
        if cusp:
            data['boundary_p'],boundary_audit=fit_boundary(data,F)
            (out/'boundary_audit.json').write_text(json.dumps(boundary_audit,indent=2)+'\n')
        np.savez_compressed(file,**data)
        (out/'flat_audit.json').write_text(json.dumps(dict(face_holonomy=audit,representation_sha256=rhohash),indent=2)+'\n')
        records=[];p=None;first=None;o=SimpleNamespace(F=F,M=dict(tz=data['tz']))
        for target in ['source','flex_1','flex_2','flex_b1','flex_b2','return']:
            label='source' if target=='return' else target
            FF,tz,warp=make_geometry(case,o,label)
            metric=solve_metric(FF,N=cfg['N'],rmin=cfg['R']/4,ntheta=cfg['ntheta'],ratio=cfg['ratio'],quad=4,cache=ROOT/'metric_cache'/f'{name}_{level}_{label}')
            p,solver=return_solve(data,start=p,tz=tz);m=measure(data,p,tz,FF,metric)
            record=dict(base=base,control=control,level=level,target=target,mesh=cfg,solver=solver,measurement=m,representation_sha256=rhohash,
                        boundary_model='fitted_parabolic_horocycle' if cusp else 'input_metric',
                        status='exact_isomonodromy_source_membership_approximate')
            if label=='source':
                G=np.array(m['hopf_gram']);G=G[...,0]+1j*G[...,1]
                c=np.array(m['hopf_coefficients']);c=c[:,0]+1j*c[:,1];d=c-intended
                record['intended_hopf_relative_error']=float(np.sqrt(np.real(d.conj()@G@d)/np.real(intended.conj()@G@intended)))
            else:record['flex_parameters']=[[float(v.real),float(v.imag)] for v in case.targets[label].u]
            if target=='source':first=p.copy()
            if target=='return':
                record['return_coordinate_max']=float(np.max(np.linalg.norm(p-first,axis=1)))
                record['return_pencil_difference']=abs(m['pencil_defect']-records[0]['measurement']['pencil_defect'])
                record['return_fibre_difference']=abs(m['normalized_fibre_defect']-records[0]['measurement']['normalized_fibre_defect'])
            np.savez_compressed(out/f'{target}.npz',p=p)
            (out/f'{target}.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
            print(name,level,target,'eps',m['pencil_defect'],'fibre',m['normalized_fibre_defect'],'eta',m['holomorphic_fit_residual'],'gradient',solver['gradient_inf'],flush=True)
        (out/'summary.json').write_text(json.dumps(dict(representation_sha256=rhohash,flatness=audit,results=records),indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='A');ap.add_argument('--control',action='store_true');ap.add_argument('--cusp',action='store_true');ap.add_argument('--levels',nargs='+',default=['F','M1','M2','C1']);a=ap.parse_args()
    run(a.base,a.control,a.levels,a.cusp)
