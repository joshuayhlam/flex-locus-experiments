"""Independent discrete-connection and original-source return checks.

Development output only. Do not interpret transport until seed/flatness and
spectral reconstruction checks converge with mesh and cusp refinement.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,newton_krylov,NoConvergence
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu,LinearOperator
from types import SimpleNamespace
from seed_higgs import Seed,P,I,adj,comm,cj,cutoff,background_gradient,exp_hermitian

ETA=np.array([1,-1,-1,-1.])
PAULI=np.concatenate((I[None],P),axis=0)

def inv(a):
    b=np.empty_like(a);b[...,0,0]=a[...,1,1];b[...,1,1]=a[...,0,0]
    b[...,0,1]=-a[...,0,1];b[...,1,0]=-a[...,1,0]
    return b/np.linalg.det(a)[...,None,None]

def exp2(a):
    a=a-.5*np.trace(a,axis1=-2,axis2=-1)[...,None,None]*I
    r2=.5*np.einsum('...ij,...ji->...',a,a);r=np.sqrt(r2+0j)
    fac=np.sinh(r)/np.maximum(abs(r),1e-300)*np.exp(-1j*np.angle(r))
    fac=np.where(abs(r)<1e-6,1+r2/6+r2*r2/120,fac)
    return np.cosh(r)[...,None,None]*I+fac[...,None,None]*a

def lorentz(g):
    b=g[...,None,:,:]@PAULI@adj(g)[...,None,:,:]
    return .5*np.einsum('aij,...bji->...ab',PAULI,b).real

def point_frame(p):
    r=np.sqrt(np.sum(p*p,axis=-1));a=np.sqrt(1+r*r)
    X=a[...,None,None]*I+np.einsum('...a,aij->...ij',p,P)
    return (X+I)/np.sqrt(2*(a+1))[...,None,None]

def hermitian_log_half(a):
    h=a@adj(a);c=np.maximum(1,.5*np.trace(h,axis1=-2,axis2=-1).real)
    d=np.arccosh(c);fac=np.where(d<1e-6,.5-d*d/12,d/(2*np.sqrt(np.maximum(c*c-1,1e-300))))
    return fac[...,None]*.5*np.einsum('aij,...ji->...a',P,h).real

def edge_data(o,steps=8):
    M=o.M;F=o.F;parent_tri=M['tri']
    if o.degree==2:
        sub=np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
        tri=parent_tri[:,sub].reshape(-1,3)
        mids=.5*(M['tz']+np.roll(M['tz'],-1,axis=1))
        allz=np.concatenate((M['tz'],mids),axis=1)
        tz=allz[:,sub].reshape(-1,3);parents=np.repeat(np.arange(len(parent_tri)),4)
        areas=np.repeat(M['area']/4,4)
    else:
        tri=parent_tri;tz=M['tz'];parents=np.arange(len(tri));areas=M['area']
    edge={}
    for t,row in enumerate(tri):
        for a,b in [(0,1),(1,2),(2,0)]:
            i,j=int(row[a]),int(row[b]);i,j=(i,j) if i<j else(j,i)
            edge.setdefault((i,j),[]).append(int(parents[t]))
    edges=np.array(list(edge));left=edges[:,0];right=edges[:,1]
    dz=F.T.center(M['z'][right]-M['z'][left])
    pp=[list(set(tt)) for tt in edge.values()]
    p0=np.array([tt[0] for tt in pp]);p1=np.array([tt[-1] for tt in pp])
    def fields(z,pa):
        dz=F.T.center(z-M['tz'][pa,0]);v=np.stack((dz.real,dz.imag),axis=-1)
        l=np.einsum('nix,nx->ni',M['grad'][pa],v);l[:,0]+=1
        if o.degree==2:
            shape=np.concatenate((l*(2*l-1),4*l[:,[0,1,2]]*l[:,[1,2,0]]),axis=1)
            g=M['grad'][pa];grad=[(4*l[:,i,None]-1)*g[:,i] for i in range(3)]
            for i,j in [(0,1),(1,2),(2,0)]:grad.append(4*(l[:,i,None]*g[:,j]+l[:,j,None]*g[:,i]))
            grad=np.stack(grad,axis=1)
        else:shape=l;grad=M['grad'][pa]
        un=o.u[parent_tri[pa]];sn=o.s[parent_tri[pa]]
        return (np.einsum('ni,ni->n',un,shape),np.einsum('ni,nix->nx',un,grad),
                np.einsum('nia,ni->na',sn,shape),np.einsum('nia,nix->nxa',sn,grad))
    norms=np.array(o.coefficients['norms']);cin=complex(*o.coefficients['kernel_correction'])
    fscale=o.coefficients['f_scaling']*o.amplitude
    links=np.broadcast_to(I,(len(edges),2,2)).copy()
    for step in range(steps):
        t=(step+.5)/steps;z=M['z'][left]+t*dz
        ff0,ff1=fields(z,p0),fields(z,p1)
        u,ug,s,sg=[.5*(a+b) for a,b in zip(ff0,ff1)]
        for p in F.p:u-=F.T.green(z-p)[0]
        grad=ug+background_gradient(F,z);uz=.5*(grad[:,0]-1j*grad[:,1])
        psi=F.differentials(z);Q=psi[:,0]/norms[0]+(.4+.3j)*psi[:,1]/norms[1]
        phi=np.zeros((len(edges),2,2),complex);phi[:,0,1]=.5*Q*np.exp(-u);phi[:,1,0]=.5*np.exp(u)
        chi,_=cutoff(F,z,M['sep'])
        f=psi[:,0].conj()-cin*psi[:,2].conj()
        if o.control:f+=.35*norms[0]/norms[2]*psi[:,2].conj()
        f=fscale*chi*f
        bc=.5*uz.conj()[:,None,None]*P[2]+f[:,None,None]*phi
        H,Hi,Hz,Hb=exp_hermitian(s,sg[:,0,:],sg[:,1,:])
        Az=Hi@Hz-Hi@adj(bc)@H+phi
        Ab=bc+Hi@adj(phi)@H
        D=(Az*dz[:,None,None]+Ab*dz.conj()[:,None,None])/steps
        links=links@exp2(D)
    frames=exp2(-.5*np.einsum('na,aij->nij',o.s,P))
    links=inv(frames[left])@links@frames[right]
    links/=np.sqrt(np.linalg.det(links))[:,None,None]
    lookup={tuple(ee):k for k,ee in enumerate(edges)}
    product=np.broadcast_to(I,(len(tri),2,2)).copy()
    for a,b in [(0,1),(1,2),(2,0)]:
        ids=np.array([lookup[tuple(sorted((int(i),int(j))))] for i,j in tri[:,[a,b]]])
        g=links[ids].copy();rev=tri[:,a]>tri[:,b];g[rev]=inv(g[rev]);product=product@g
    err=np.linalg.norm(product-I,axis=(1,2))
    # A scale-sensitive flatness diagnostic; raw small-face holonomy alone
    # would shrink for nonflat connections and is not a consistency check.
    Umean=o.U.mean(axis=1)[parents]
    curv=np.sqrt(np.sum(err**2/areas*np.exp(-2*Umean)))
    return dict(edges=edges,links=links,dz=dz,triangle_holonomy=err,curvature_proxy=float(curv),
                area=areas,tri=tri,tz=tz,z=M['z'],free=o.free,node_U=o.u+M['u0nodes'])

def weights(data,tz=None):
    if tz is None:tz=data['tz']
    tri=data['tri'];lookup={tuple(ee):k for k,ee in enumerate(data['edges'])};out=np.zeros(len(lookup))
    for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
        u=tz[:,a]-tz[:,c];v=tz[:,b]-tz[:,c]
        cot=(u*v.conj()).real/abs((u*v.conj()).imag)
        ids=np.array([lookup[tuple(sorted((int(i),int(j))))] for i,j in tri[:,[a,b]]])
        np.add.at(out,ids,.5*cot)
    return out

def return_solve(data,maxiter=3000,start=None,tz=None):
    ij=data['edges'];i,j=ij.T;free=data['free'];nn=len(data['z']);w=weights(data,tz)
    rel=lorentz(data['links'])
    def points(v):
        x=np.zeros((nn,4));x[:,0]=1
        if 'boundary_p' in data:
            x[:,1:]=data['boundary_p'];x[:,0]=np.sqrt(1+np.sum(x[:,1:]**2,axis=1))
        x[free,1:]=v.reshape(-1,3);x[free,0]=np.sqrt(1+np.sum(x[free,1:]**2,axis=1));return x
    def fun(v):
        x=points(v);left=x[i];right=np.einsum('nij,nj->ni',rel,x[j]);C=np.maximum(1,np.sum(left*right*ETA,axis=1));d=np.arccosh(C)
        f=np.where(d<1e-6,1-d*d/6,d/np.sqrt(np.maximum(C*C-1,1e-300)))*w
        li=right*ETA;lj=np.einsum('nij,ni->nj',rel,left*ETA)
        gi=f[:,None]*(li[:,1:]+li[:,0,None]*x[i,1:]/x[i,0,None]);gj=f[:,None]*(lj[:,1:]+lj[:,0,None]*x[j,1:]/x[j,0,None])
        g=np.stack([np.bincount(i,weights=gi[:,k],minlength=nn)+np.bincount(j,weights=gj[:,k],minlength=nn) for k in range(3)],axis=-1)
        return .5*np.sum(w*d*d),g[free].ravel()
    v=np.zeros(len(free)*3) if start is None else start[free].ravel()
    e0,g0=fun(v);starttime=time.time()
    rng=np.random.default_rng(4616);vv=rng.normal(size=len(v));vv/=np.linalg.norm(vv);delta=1e-6
    fd=(fun(v+delta*vv)[0]-fun(v-delta*vv)[0])/(2*delta)
    gradient_check_error=float(abs(fd-g0@vv))
    diag=np.bincount(i,weights=w,minlength=nn)+np.bincount(j,weights=w,minlength=nn)
    potential=np.abs(w)*np.maximum(rel[:,0,0]-1,0)
    diag+=np.bincount(i,weights=potential,minlength=nn)+np.bincount(j,weights=potential,minlength=nn)
    K=coo_matrix((np.r_[-w,-w],(np.r_[i,j],np.r_[j,i])),shape=(nn,nn)).tocsr()+diags(diag)
    lu=splu(K[free][:,free].tocsc());op=LinearOperator((len(v),len(v)),matvec=lambda r:lu.solve(r.reshape(-1,3)).ravel())
    nh=[]
    try:
        vv=newton_krylov(lambda x:fun(x)[1],v,inner_M=op,f_tol=2e-10,maxiter=20,inner_maxiter=40,
                         callback=lambda x,f:nh.append(float(abs(f).max())))
        en,gr=fun(vv)
        if en>e0+1e-9:raise NoConvergence(vv)
        res=SimpleNamespace(x=vv,fun=en,jac=gr,success=True,message='Newton-Krylov gradient tolerance met',nit=len(nh))
    except (NoConvergence,ValueError) as exc:
        warm=exc.args[0] if isinstance(exc,NoConvergence) else v
        res=minimize(fun,warm,jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter,gtol=1e-9,ftol=1e-15,maxcor=30,maxls=30))
    p=points(res.x)[:,1:];distance=np.arcsinh(np.linalg.norm(p,axis=1))
    return p,dict(success=bool(res.success),message=str(res.message),iterations=res.nit,energy_initial=e0,energy=float(res.fun),
        gradient_inf=float(abs(res.jac).max()),initial_gradient_inf=float(abs(g0).max()),gradient_check_error=gradient_check_error,
        return_point_distance_max=float(max(distance)),return_point_distance_rms=float(np.sqrt(np.mean(distance**2))),
        negative_weights=int(sum(w<-1e-12)),newton_gradient_history=nh,seconds=time.time()-starttime)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='A');ap.add_argument('--N',type=int,default=12)
    ap.add_argument('--R',type=float,default=.002);ap.add_argument('--amplitude',type=float,default=.04);ap.add_argument('--control',action='store_true');ap.add_argument('--coeff');ap.add_argument('--steps',type=int,default=8)
    a=ap.parse_args();o=Seed(a.base,a.N,a.R,amplitude=a.amplitude,control=a.control,coeff_path=a.coeff);seed=o.solve()
    d=edge_data(o,a.steps);p,ret=return_solve(d)
    ans=dict(status='prototype_not_validated',seed=seed,return_check=ret,flatness=dict(curvature_proxy=d['curvature_proxy'],face_holonomy_max=float(max(d['triangle_holonomy'])),face_holonomy_median=float(np.median(d['triangle_holonomy']))))
    out=Path(__file__).parent/'checks';out.mkdir(exist_ok=True)
    name=f'{a.base}_{"control" if a.control else "fibre"}_P{o.degree}_N{a.N}_R{a.R}_b{a.amplitude}'
    (out/(name+'.json')).write_text(json.dumps(ans,indent=2)+'\n');np.savez_compressed(out/(name+'.npz'),p=p,links=d['links'],edges=d['edges'],s=o.s,u=o.u,z=d['z'],tz=d['tz'],tri=d['tri'])
    print(json.dumps(ans,indent=2))
