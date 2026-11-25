"""Repeat the original six inverse solves, exclusively on target stage F1."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,hashlib,time
import numpy as np
import flex_wp as fw
from seed_higgs import BASES,cj
from run_transport import make_geometry
from geometry_inputs import make_cases
from target_metric import solve_metric
from connection_check import return_solve
from flat_interpolate import face_errors
from spectral_measure import measure
from cusp_boundary import fit_boundary
from refine_forward import CONFIG

ROOT=Path(__file__).parent
TARGETS=['source','flex_1','flex_2','flex_b1','flex_b2','return']

def run(base,control,stage,source_only=False):
    name=base+('_control' if control else '_fibre');forward=ROOT/'forward'/f'{name}_{stage}'
    out=ROOT/'runs'/f'{name}_{stage}_F1';out.mkdir(parents=True,exist_ok=True)
    with np.load(forward/'flat_connection.npz') as v:data={k:v[k] for k in v.files}
    rhohash=hashlib.sha256(data['generators'].tobytes()).hexdigest()
    F=fw.Flex(*BASES[base]);case=make_cases(include_identity=False)[base]
    cfg=CONFIG['J0'];audit=face_errors(data);assert audit['maximum']<1e-8,audit
    data['boundary_p'],boundary_audit=fit_boundary(data,F)
    np.savez_compressed(out/'connection.npz',**data)
    (out/'boundary_audit.json').write_text(json.dumps(boundary_audit,indent=2))
    (out/'flat_audit.json').write_text(json.dumps(dict(face_holonomy=audit,representation_sha256=rhohash),indent=2))
    coef=json.loads((ROOT/'inputs'/f'{base}_coefficients.json').read_text())
    intended=np.array([1/coef['norms'][0],(.4+.3j)/coef['norms'][1],0j])
    seed=json.loads((ROOT/'canonical'/f'{name}_N64'/'seed.json').read_text())
    vv=np.array(seed['serre_pairings']);expected=2*(vv[:,0]+1j*vv[:,1])
    records=[];p=None;first=None;o=SimpleNamespace(F=F,M=dict(tz=data['tz']))
    for target in TARGETS[:1] if source_only else TARGETS:
        label='source' if target=='return' else target
        filename=out/f'{target}.json'
        if filename.exists():
            r=json.loads(filename.read_text());assert r['representation_sha256']==rhohash
            p=np.load(out/f'{target}.npz')['p'];records.append(r)
            if target=='source':first=p.copy()
            continue
        FF,tz,warp=make_geometry(case,o,label)
        metric=solve_metric(FF,N=64,rmin=6.25e-5/4,ntheta=80,ratio=1.18,quad=4,
            cache=ROOT/'metric_cache'/f'{base}_fibre_F_{label}')
        p,solver=return_solve(data,start=p,tz=tz)
        assert solver['success'] and solver['gradient_inf']<1e-7,solver
        m=measure(data,p,tz,FF,metric)
        r=dict(base=base,control=control,forward_stage=stage,target_stage='F1',target=target,
            forward_mesh=CONFIG[stage],target_mesh=cfg,solver=solver,measurement=m,
            representation_sha256=rhohash,boundary_model='fitted_parabolic_horocycle')
        if label=='source':
            G=np.array(m['hopf_gram']);G=G[...,0]+1j*G[...,1]
            cc=np.array(m['hopf_coefficients']);cc=cc[:,0]+1j*cc[:,1];delta=cc-intended
            r['intended_hopf_relative_error']=float(np.sqrt(np.real(delta.conj()@G@delta)/np.real(intended.conj()@G@intended)))
            observed=np.array(m['all_prym_pairings']);observed=observed[:,0]+1j*observed[:,1]
            norms=np.array(m['all_prym_form_norms'])
            r['source_prym_audit']=dict(expected=cj(expected),observed=cj(observed),
                componentwise_normalized_error=(abs(observed-expected)/norms).tolist(),
                relative_normalized_vector_error=float(np.linalg.norm((observed-expected)/norms)/np.linalg.norm(expected/norms)))
        else:r['flex_parameters']=[[float(v.real),float(v.imag)] for v in case.targets[label].u]
        if target=='source':first=p.copy()
        if target=='return':
            r['return_coordinate_max']=float(np.max(np.linalg.norm(p-first,axis=1)))
            r['return_pencil_difference']=abs(m['pencil_defect']-records[0]['measurement']['pencil_defect'])
            r['return_fibre_difference']=abs(m['normalized_fibre_defect']-records[0]['measurement']['normalized_fibre_defect'])
        np.savez_compressed(out/f'{target}.npz',p=p);filename.write_text(json.dumps(r,indent=2));records.append(r)
        print(name,stage,target,'eps',m['pencil_defect'],'fibre',m['normalized_fibre_defect'],'eta',m['holomorphic_fit_residual'],
            'hopf_error',r.get('intended_hopf_relative_error'),'prym_error',r.get('source_prym_audit',{}).get('relative_normalized_vector_error'),flush=True)
        if target=='source' and not control:
            # Diagnostic only: the unflattened connection is not used for
            # isomonodromy. In its solved unitary frame the intended map is 0.
            with np.load(forward/'raw_connection.npz') as v:raw={k:v[k] for k in v.files}
            rm=measure(raw,np.zeros_like(p),raw['tz'],F,metric)
            GG=np.array(rm['hopf_gram']);GG=GG[...,0]+1j*GG[...,1]
            cc=np.array(rm['hopf_coefficients']);cc=cc[:,0]+1j*cc[:,1];delta=cc-intended
            rv=np.array(rm['all_prym_pairings']);rv=rv[:,0]+1j*rv[:,1];rn=np.array(rm['all_prym_form_norms'])
            diagnostic=dict(purpose='F1 extraction of the unflattened forward connection at its prescribed unitary metric; not a representation test',
                measurement=rm,intended_hopf_relative_error=float(np.sqrt(np.real(delta.conj()@GG@delta)/np.real(intended.conj()@GG@intended))),
                relative_normalized_prym_error=float(np.linalg.norm((rv-expected)/rn)/np.linalg.norm(expected/rn)))
            (out/'raw_source_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
            print(name,stage,'RAW EXTRACTION',diagnostic['intended_hopf_relative_error'],diagnostic['relative_normalized_prym_error'],flush=True)
    (out/('source_summary.json' if source_only else 'summary.json')).write_text(json.dumps(dict(results=records,representation_sha256=rhohash,flatness=audit),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',choices=['A','B'],required=True);ap.add_argument('--control',action='store_true');ap.add_argument('--stage',choices=CONFIG,required=True);ap.add_argument('--source-only',action='store_true')
    a=ap.parse_args();run(a.base,a.control,a.stage,a.source_only)
