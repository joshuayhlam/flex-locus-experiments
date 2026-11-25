"""Prototype generic-source Higgs metric solver; NOT validated experiment data.

Build section-relative spectral-line deformations by commuting Dolbeault
perturbations. The Serre pairing detects their extra elliptic component.
The nonlinear solver must pass independent checks before transport is used.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, splu, LinearOperator
from scipy.optimize import newton_krylov, NoConvergence
import flex_wp as fw

I=np.eye(2,dtype=complex)
P=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
BASES={'A':(.23+1.08j,5+2j),'B':(.07+.94j,-4+3j)}

def adj(x): return np.swapaxes(x.conj(),-1,-2)
def comm(x,y): return x@y-y@x
def cj(x):
    a=np.asarray(x); return np.stack((a.real,a.imag),axis=-1).tolist()

def cutoff(F,z,sep):
    """Smooth, periodic, identically zero near each puncture."""
    chi=np.ones(z.shape); grad=np.zeros(z.shape+(2,))
    for p,d in zip(F.p,sep):
        v=F.T.center(z-p); r=abs(v); r0=.08*d; width=.12*d
        t=np.clip((r-r0)/width,0,1)
        a=10*t**3-15*t**4+6*t**5
        ar=(30*t**2-60*t**3+30*t**4)/width
        ag=ar[...,None]*np.stack((v.real,v.imag),axis=-1)/np.maximum(r,1e-300)[...,None]
        grad=grad*a[...,None]+chi[...,None]*ag
        chi*=a
    return chi,.5*(grad[...,0]-1j*grad[...,1])

def background_gradient(F,z):
    g=np.zeros(z.shape+(2,))
    for p in F.p:g-=F.T.green(z-p)[1]
    return g

def exp_hermitian(s,sx,sy):
    r2=np.sum(s*s,axis=-1);r=np.sqrt(r2)
    small=r<1e-5
    shc=np.sinh(r)/np.maximum(r,1e-300)
    b=(r*np.cosh(r)-np.sinh(r))/np.maximum(r**3,1e-300)
    shc=np.where(small,1+r2/6+r2*r2/120,shc)
    b=np.where(small,1/3+r2/30+r2*r2/840,b)
    S=np.einsum('...a,aij->...ij',s,P)
    H=np.cosh(r)[...,None,None]*I+shc[...,None,None]*S
    Hi=np.cosh(r)[...,None,None]*I-shc[...,None,None]*S
    def der(ds):
        dot=np.sum(s*ds,axis=-1)
        return (shc*dot)[...,None,None]*I+shc[...,None,None]*np.einsum('...a,aij->...ij',ds,P)+(b*dot)[...,None,None]*S
    Hx,Hy=der(sx),der(sy)
    return H,Hi,.5*(Hx-1j*Hy),.5*(Hx+1j*Hy)

def scalar_seed(F,M,Q):
    M['qabs2']=abs(Q)**2
    dist=np.min([abs(F.T.center(M['z']-p)) for p in F.p],axis=0)
    u=-np.log(dist)-np.log(.8-np.log(dist))+.4-M['u0nodes']
    hist=[]
    for it in range(40):
        E,r,H,area,tail=fw.nonlinear(M,u)
        hist.append(float(abs(r).max()))
        if abs(r).max()<1e-11:break
        du=spsolve(H,-r);step=1.
        while step>1e-8:
            ne,*_=fw.nonlinear(M,u+step*du,False)
            if ne<=E+1e-4*step*(r@du)+1e-12:break
            step*=.5
        u+=step*du
    return u,hist

class Seed:
    def __init__(self,base,N=12,R=.002,ntheta=20,quad=4,amplitude=.04,control=False,coeff_path=None,degree=2,ratio=1.35,exact_gauge=False):
        self.base,self.N,self.R,self.amplitude,self.control=base,N,R,amplitude,control
        fw.quadrature_order(quad)
        self.F=F=fw.Flex(*BASES[base]);self.M=M=fw.mesh(F,N,R,ntheta,ratio)
        self.ntheta,self.ratio,self.quad=ntheta,ratio,quad
        self.degree=degree
        if degree==2:fw.quadratic_elements(F,M)
        fw.cusp_background(F,M,'green')
        self.z=M['tz']@fw.BARY.T
        self.weight=M['area'][:,None]*fw.BW
        self.shape=M.get('shape',fw.BARY).copy()
        self.gradq=M.get('gradq',np.broadcast_to(M['grad'][:,None],(len(M['tri']),len(fw.BW),3,2)))
        self.psi=F.differentials(self.z)
        chi,chiz=cutoff(F,self.z,M['sep'])
        pair=np.einsum('tq,tqi,tqj->ij',self.weight*chi,self.psi,self.psi.conj())
        if coeff_path:
            pars=json.loads(Path(coeff_path).read_text())
            norms=np.array(pars['norms'])
            cin=complex(*pars['kernel_correction'])
            scaling=pars['f_scaling']
        else:
            norms=np.sqrt(np.diag(pair).real)
            cin=pair[2,0]/pair[2,2]
            q0=(self.psi[...,0]/norms[0]+(.4+.3j)*self.psi[...,1]/norms[1])/4
            ff=chi*(self.psi[...,0].conj()-cin*self.psi[...,2].conj())
            scaling=1/np.sqrt(2*np.sum(self.weight*abs(q0)*abs(ff)**2))
        self.coefficients=dict(norms=norms.tolist(),kernel_correction=cj(cin),f_scaling=float(scaling))
        self.Q=self.psi[...,0]/norms[0]+(.4+.3j)*self.psi[...,1]/norms[1]
        fbase=self.psi[...,0].conj()-cin*self.psi[...,2].conj()
        if control:fbase+=.35*norms[0]/norms[2]*self.psi[...,2].conj()
        self.f=amplitude*scaling*chi*fbase
        fz=amplitude*scaling*chiz*fbase
        self.exact_gauge=exact_gauge
        if exact_gauge:
            grid=(np.arange(17)+.5)/17
            zz=F.T.z(np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2))
            sep=np.min([abs(F.T.center(zz-p)) for p in F.p],axis=0)
            self.gauge_center=zz[np.argmax(sep)];radius=.16
            v=F.T.center(self.z-self.gauge_center);rr=abs(v)**2/radius**2;tt=np.maximum(1-rr,0)
            self.f=-4*amplitude/radius**2*tt**3*v
            fz=-4*amplitude/radius**2*tt**3+12*amplitude/radius**4*abs(v)**2*tt**2
        self.pairings=np.einsum('tq,tq,tqi->i',self.weight,self.f,self.psi)
        self.u,self.scalar_history=scalar_seed(F,M,self.Q)
        self.U=self.u[M['tri']]@self.shape.T+M['u0q']
        ug=np.einsum('ti,tqix->tqx',self.u[M['tri']],self.gradq)+background_gradient(F,self.z)
        Uz=.5*(ug[...,0]-1j*ug[...,1]);self.Uz=Uz
        d=1e-6
        Qp=(F.differentials(self.z+d)-F.differentials(self.z-d))/(2*d)
        Qp=Qp[...,0]/norms[0]+(.4+.3j)*Qp[...,1]/norms[1]
        self.phi=phi=np.zeros(self.z.shape+(2,2),complex)
        phi[...,0,1]=.5*self.Q*np.exp(-self.U)
        phi[...,1,0]=.5*np.exp(self.U)
        dzphi=np.zeros_like(phi)
        dzphi[...,0,1]=.5*np.exp(-self.U)*(Qp-2*self.Q*Uz)
        dzphi[...,1,0]=2*Uz*phi[...,1,0]
        self.B=self.f[...,None,None]*phi
        deriv=fz[...,None,None]*phi+self.f[...,None,None]*dzphi
        self.M0=deriv+adj(deriv)+comm(self.B,adj(self.B))
        self.Bc=.5*Uz.conj()[...,None,None]*P[2]+self.B
        self.free=np.setdiff1d(np.arange(M['nv']),np.unique(M['be']))
        massval=np.sum(abs(phi)**2,axis=(-1,-2))
        ml=np.einsum('tq,qi,qj->tij',self.weight*massval,self.shape,self.shape)
        pre=.25*M['K']+coo_matrix((ml.ravel(),(M['rows'],M['cols'])),shape=M['K'].shape).tocsr()
        self.pre=splu(pre[self.free][:,self.free].tocsc())
        self.calls=0

    def exact_gauge_log_metric(self):
        z=self.M['z'];U=self.u+self.M['u0nodes'];norms=np.array(self.coefficients['norms'])
        psi=self.F.differentials(z);Q=psi[:,0]/norms[0]+(.4+.3j)*psi[:,1]/norms[1]
        phi=np.zeros((len(z),2,2),complex);phi[:,0,1]=.5*Q*np.exp(-U);phi[:,1,0]=.5*np.exp(U)
        chi=self.amplitude*np.maximum(1-abs(self.F.T.center(z-self.gauge_center))**2/.16**2,0)**4
        lam=.5*np.sqrt(Q);v=chi*lam
        g=np.cosh(v)[:,None,None]*I+(np.sinh(v)/lam)[:,None,None]*phi
        H=adj(g)@g;c=.5*np.trace(H,axis1=-2,axis2=-1).real;r=np.arccosh(np.maximum(c,1))
        factor=np.where(r<1e-7,1-r*r/6,r/np.sqrt(np.maximum(c*c-1,1e-300)))
        return .5*factor[:,None]*np.einsum('aij,nji->na',P,H).real

    def residual(self,v):
        M=self.M; s=np.zeros((M['nv'],3));s[self.free]=v.reshape(-1,3)
        sq=np.einsum('tia,qi->tqa',s[M['tri']],self.shape)
        sg=np.einsum('tia,tqix->tqxa',s[M['tri']],self.gradq)
        H,Hi,Hz,Hb=exp_hermitian(sq,sg[:,:,0,:],sg[:,:,1,:])
        C=Hi@(Hz-comm(adj(self.Bc),H))
        val=H@(self.M0+comm(self.phi,Hi@adj(self.phi)@H-adj(self.phi))+comm(C,self.Bc))+Hb@C
        flux=H@C
        valp=.5*np.einsum('aij,tqji->tqa',P,val).real
        fluxp=.5*np.einsum('aij,tqji->tqa',P,flux)
        gb=.5*(self.gradq[...,0]+1j*self.gradq[...,1])
        local=np.einsum('tq,tqa,qi->tia',self.weight,valp,self.shape)
        local+=np.einsum('tq,tqa,tqi->tia',self.weight,fluxp,gb).real
        out=np.stack([np.bincount(M['tri'].ravel(),weights=local[...,a].ravel(),minlength=M['nv']) for a in range(3)],axis=-1)
        self.calls+=1
        return out[self.free].ravel()

    def solve(self,maxiter=30):
        size=len(self.free)*3;v=np.zeros(size);hist=[];start=time.time()
        def pre(r):return self.pre.solve(r.reshape(-1,3)).ravel()
        op=LinearOperator((size,size),matvec=pre)
        def cb(x,f):
            hist.append(dict(residual_inf=float(abs(f).max()),log_metric_max=float(abs(x).max()),seconds=time.time()-start))
            print(self.base,self.N,self.control,len(hist),hist[-1],flush=True)
        try:
            v=newton_krylov(self.residual,v,inner_M=op,f_tol=3e-10,maxiter=maxiter,inner_maxiter=40,callback=cb)
            success=True
        except NoConvergence as e:
            v=e.args[0];success=False
        res=self.residual(v)
        s=np.zeros((self.M['nv'],3));s[self.free]=v.reshape(-1,3)
        self.s=s
        return dict(status='prototype_not_validated',base=self.base,N=self.N,R=self.R,control=self.control,degree=self.degree,
            ntheta=self.ntheta,ratio=self.ratio,quadrature=self.quad,
            amplitude=self.amplitude,nodes=self.M['nv'],success=success,residual_inf=float(abs(res).max()),
            scalar_residual=self.scalar_history[-1],serre_pairings=cj(self.pairings),coefficients=self.coefficients,
            history=hist,seconds=time.time()-start,residual_calls=self.calls,log_metric_max=float(abs(s).max()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',choices=list(BASES),default='A');ap.add_argument('--N',type=int,default=12)
    ap.add_argument('--R',type=float,default=.002);ap.add_argument('--amplitude',type=float,default=.04);ap.add_argument('--control',action='store_true')
    ap.add_argument('--coeff');ap.add_argument('--maxiter',type=int,default=30);ap.add_argument('--out',default='prototype')
    a=ap.parse_args();o=Seed(a.base,a.N,a.R,amplitude=a.amplitude,control=a.control,coeff_path=a.coeff)
    ans=o.solve(a.maxiter);p=Path(__file__).parent/a.out;p.mkdir(exist_ok=True)
    name=f'{a.base}_{"control" if a.control else "fibre"}_N{a.N}_R{a.R}_b{a.amplitude}'
    (p/(name+'.json')).write_text(json.dumps(ans,indent=2)+'\n')
    np.savez_compressed(p/(name+'.npz'),s=o.s,u=o.u,z=o.M['z'],tri=o.M['tri'],tz=o.M['tz'])
    print(json.dumps(ans,indent=2))
