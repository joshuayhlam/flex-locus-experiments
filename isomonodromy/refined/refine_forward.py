"""Refine the forward solve and integrate on the fixed F1 cellular complex."""
from pathlib import Path
import argparse,json,time,hashlib
from types import SimpleNamespace
import numpy as np
import flex_wp as fw
from seed_higgs import Seed,P,cj
from path_holonomy import Field
from connection_check import I,inv,exp2
from flat_bundle import reconstruct

ROOT=Path(__file__).parent
CONFIG={'J0':dict(N=64,R=6.25e-5,ntheta=80,ratio=1.18),
        'J1':dict(N=96,R=1.5625e-5,ntheta=112,ratio=1.14),
        'J2':dict(N=128,R=1.5625e-5,ntheta=112,ratio=1.14)}

def evaluate_nodal(H,z,array):
    pa=H.locate(z);l=H.bary(z,pa)
    shape=np.c_[l*(2*l-1),4*l[:,[0,1,2]]*l[:,[1,2,0]]]
    return np.einsum('ni,ni...->n...',shape,array[H.M['tri'][pa]])

def split_edges(H,a,b):
    delta=b-a;row=np.arange(len(a));lo=np.zeros(len(a));hi=np.ones(len(a));parts=[];snap=0.
    for iteration in range(100):
        if not len(row):break
        pa=H.locate(a[row]+.5*(lo+hi)*delta[row])
        tiny=(hi-lo)*abs(delta[row])<1e-10
        if np.any(tiny):
            snap+=float(np.sum((hi[tiny]-lo[tiny])*abs(delta[row[tiny]])))
            parts.append((row[tiny],lo[tiny],hi[tiny],pa[tiny]))
            row,lo,hi,pa=[x[~tiny] for x in [row,lo,hi,pa]]
            if not len(row):break
        la=H.bary(a[row]+lo*delta[row],pa)
        length=(hi-lo)*delta[row]
        dl=np.einsum('nix,nx->ni',H.M['grad'][pa],np.c_[length.real,length.imag])
        with np.errstate(divide='ignore',invalid='ignore'):tt=-la/dl
        left=np.clip(np.max(np.where(dl>1e-14,tt,-np.inf),axis=1),0,1)
        right=np.clip(np.min(np.where(dl<-1e-14,tt,np.inf),axis=1),0,1)
        l=lo+left*(hi-lo);r=lo+right*(hi-lo)
        parts.append((row.copy(),l,r,pa))
        tl=l-lo>1e-10;tr=hi-r>1e-10
        row,lo,hi=np.r_[row[tl],row[tr]],np.r_[lo[tl],r[tr]],np.r_[l[tl],hi[tr]]
    if len(row):raise RuntimeError('Cell splitting failed')
    row,lo,hi,pa=[np.concatenate([p[j] for p in parts]) for j in range(4)]
    order=np.lexsort((lo,row));row,lo,hi,pa=[x[order] for x in [row,lo,hi,pa]]
    coverage=np.bincount(row,weights=hi-lo,minlength=len(a));err=float(max(abs(coverage-1)))
    assert err<1e-7 and snap<1e-7,(err,snap)
    return row,a[row]+lo*delta[row],a[row]+hi*delta[row],pa,dict(coverage_error=err,snapped_length=snap)

def integrate_parts(H,row,a,b,pa,count,subdivide):
    dz=b-a
    if subdivide>1:
        t=np.arange(subdivide)/subdivide
        aa=(a[:,None]+dz[:,None]*t).ravel();bb=aa+np.repeat(dz/subdivide,subdivide)
        row,pa=np.repeat(row,subdivide),np.repeat(pa,subdivide);a,b=aa,bb;dz=b-a
    c1=.5-np.sqrt(3)/6;c2=.5+np.sqrt(3)/6
    az1,ab1=H.connection(a+c1*dz,pa);az2,ab2=H.connection(a+c2*dz,pa)
    A1=az1*dz[:,None,None]+ab1*dz.conj()[:,None,None]
    A2=az2*dz[:,None,None]+ab2*dz.conj()[:,None,None]
    alpha=(3-2*np.sqrt(3))/12;beta=(3+2*np.sqrt(3))/12
    terms=exp2(beta*A1+alpha*A2)@exp2(alpha*A1+beta*A2)
    out=np.broadcast_to(I,(count,2,2)).copy()
    sizes=np.bincount(row,minlength=count);heads=np.r_[0,np.cumsum(sizes)[:-1]]
    for k in range(max(sizes)):
        ids=np.flatnonzero(sizes>k);out[ids]=out[ids]@terms[heads[ids]+k]
    return out

def build(base,control,stage):
    name=base+('_control' if control else '_fibre');out=ROOT/'forward'/f'{name}_{stage}';out.mkdir(parents=True,exist_ok=True)
    if (out/'flat_connection.npz').exists():return out
    cfg=CONFIG[stage];old=ROOT/'canonical'/f'{name}_N64'
    scalar_u=None
    candidate=ROOT/'forward'/f'{base}_fibre_{stage}'/'solution.npz'
    if control and candidate.exists():
        # The fibre control has exactly the same Q, mesh and scalar equation.
        with np.load(candidate) as v:scalar_u=v['u'].copy()
    o=Seed(base,**cfg,quad=4,control=control,coeff_path=ROOT/'inputs'/f'{base}_coefficients.json',scalar_u=scalar_u)
    if (out/'solution.npz').exists():
        v=np.load(out/'solution.npz');o.u=v['u'];o.s=v['s'];seed=json.loads((out/'seed.json').read_text())
    elif stage=='J0':
        v=np.load(old/'raw_connection.npz');o.u=v['u'];o.s=v['s'];seed=json.loads((old/'seed.json').read_text())
        np.savez_compressed(out/'solution.npz',u=o.u,s=o.s);(out/'seed.json').write_text(json.dumps(seed,indent=2))
    else:
        warm=ROOT/'forward'/f'{name}_J1'
        use=stage=='J2' and (warm/'solution.npz').exists()
        wc=CONFIG['J1'] if use else CONFIG['J0']
        M=fw.mesh(o.F,wc['N'],wc['R'],wc['ntheta'],wc['ratio']);fw.quadratic_elements(o.F,M)
        H0=Field(SimpleNamespace(F=o.F,M=M));v=np.load(warm/'solution.npz' if use else old/'raw_connection.npz')
        z=o.F.T.z(o.F.T.uv(o.M['z'])%1);ix=H0.delaunay.find_simplex(np.c_[z.real,z.imag]);pa=H0.parent[np.maximum(ix,0)]
        good=(ix>=0)&(pa>=0);initial=np.zeros((len(z),3))
        initial[good]=evaluate_nodal(H0,z[good],v['s'])
        seed=o.solve(initial=initial);assert seed['success'] and seed['residual_inf']<1e-8
        (out/'seed.json').write_text(json.dumps(seed,indent=2));np.savez_compressed(out/'solution.npz',u=o.u,s=o.s)
        del H0,M,initial,v
    H=Field(o)
    with np.load(old/'raw_connection.npz') as v:
        data={k:v[k] for k in ['edges','tri','tz','z','free','node_U']}
    # Fix the cell complex AND the tree weights across forward refinements.
    # The gauge frames below are evaluated from the newly solved metric.
    z=data['z'];sn=evaluate_nodal(H,z,o.s)
    frames=exp2(-.5*np.einsum('na,aij->nij',sn,P));edge=data['edges'];links=[];coarse_links=[];checks=[];integration_delta=[]
    subdivision=4 if stage=='J2' else 2
    started=time.time()
    for start in range(0,len(edge),3000):
        ee=edge[start:start+3000];a=z[ee[:,0]];b=a+o.F.T.center(z[ee[:,1]]-a)
        row,aa,bb,pa,check=split_edges(H,a,b)
        coarse=integrate_parts(H,row,aa,bb,pa,len(ee),subdivision//2)
        fine=integrate_parts(H,row,aa,bb,pa,len(ee),subdivision)
        change=np.linalg.norm(fine-coarse,axis=(1,2))/(1+np.linalg.norm(fine,axis=(1,2)))
        integration_delta.extend(change.tolist())
        fine=inv(frames[ee[:,0]])@fine@frames[ee[:,1]]
        fine/=np.sqrt(np.linalg.det(fine))[:,None,None]
        coarse=inv(frames[ee[:,0]])@coarse@frames[ee[:,1]]
        coarse/=np.sqrt(np.linalg.det(coarse))[:,None,None]
        links.append(fine);coarse_links.append(coarse);checks.append(check)
        if start%12000==0:print(name,stage,'integrated',start+len(ee),'/',len(edge),flush=True)
    data['links']=np.concatenate(links)
    np.savez_compressed(out/'raw_connection.npz',**data)
    fixed,audit,extra=reconstruct(data,o.F)
    audit['integration']=dict(max_relative_edge_change=float(max(integration_delta)),rms_relative_edge_change=float(np.sqrt(np.mean(np.array(integration_delta)**2))),
        maximum_coverage_error=max(c['coverage_error'] for c in checks),intersection_roundoff_length=sum(c['snapped_length'] for c in checks),seconds=time.time()-started,
        subdivision=subdivision,comparison_subdivision=subdivision//2,
        method='Fourth-order commutator-free Magnus; original P2 cell boundaries')
    coarse_data=dict(data,links=np.concatenate(coarse_links))
    _,ca,ce=reconstruct(coarse_data,o.F)
    def character(G):
        from itertools import combinations
        values=[]
        for n in range(1,5):
            for ids in combinations(range(4),n):
                g=I.copy()
                for i in ids:g=g@G[i]
                values.append(np.trace(g))
        return np.array(values)
    fine_character=character(extra['generators']);coarse_character=character(ce['generators'])
    audit['integration']['reconstructed_character_change']=float(np.max(abs(fine_character-coarse_character)/(1+abs(fine_character))))
    audit['integration']['coarse_projection']=ca
    audit['stage']=stage;audit['forward_mesh']=cfg;audit['target_stage']='F1'
    (out/'flat_projection.json').write_text(json.dumps(audit,indent=2))
    np.savez_compressed(out/'flat_connection.npz',**{k:data[k] for k in ['edges','tri','tz','z','free','node_U']},links=fixed,frames=extra['frames'],generators=extra['generators'])
    print('FORWARD DONE',name,stage,audit['initial_peripheral_trace_error'],audit['generator_change'],flush=True)
    return out

if __name__=='__main__':
    import fcntl
    ap=argparse.ArgumentParser();ap.add_argument('--base',choices=['A','B'],required=True);ap.add_argument('--control',action='store_true');ap.add_argument('--stage',choices=CONFIG,required=True)
    a=ap.parse_args()
    with (ROOT/'forward_memory.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        build(a.base,a.control,a.stage)
