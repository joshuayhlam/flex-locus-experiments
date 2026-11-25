"""Independent punctured-torus WP test for the gothic pencil.

No Betti equations or vanishing conditions enter the metric solve.
Periodic P1 finite elements, circular cusp truncation, nonlinear cusp Robin
condition. Quadratic differentials are evaluated directly in Weierstrass form.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def cjson(z):
    return [float(np.real(z)), float(np.imag(z))]


class Torus:
    def __init__(self, tau):
        self.tau = complex(tau)
        self.A = self.tau.imag
        Q = np.exp(2j*np.pi*self.tau)
        sums = [sum(n**k * Q**n / (1-Q**n) for n in range(1,80)) for k in (1,3,5)]
        self.E2, E4, E6 = 1-24*sums[0], 1+240*sums[1], 1-504*sums[2]
        self.g2 = 4*np.pi**4/3*E4
        self.g3 = 8*np.pi**6/27*E6

    def uv(self,z):
        v=np.imag(z)/self.A
        return np.stack((np.real(z)-self.tau.real*v,v),axis=-1)

    def z(self,uv):
        return uv[...,0]+self.tau*uv[...,1]

    def center(self,z):
        uv=self.uv(z); return self.z(uv-np.floor(uv+.5))

    def wp(self,z):
        z=self.center(np.asarray(z,dtype=complex))
        th=[np.zeros_like(z) for _ in range(4)]
        for n in range(13):
            a=(2*n+1)*np.pi
            b=2*(-1)**n*np.exp(1j*np.pi*self.tau*(n+.5)**2)
            for k in range(4):
                th[k]+=b*a**k*np.sin(a*z+k*np.pi/2)
        d=th[1]/th[0]
        x=d*d-th[2]/th[0]-np.pi**2*self.E2/3
        y=-th[3]/th[0]+3*th[2]*th[1]/th[0]**2-2*d**3
        return x,y

    def green(self,z):
        z=self.center(np.asarray(z,dtype=complex));th=np.zeros_like(z);dt=np.zeros_like(z)
        for n in range(13):
            a=(2*n+1)*np.pi;b=2*(-1)**n*np.exp(1j*np.pi*self.tau*(n+.5)**2)
            th+=b*np.sin(a*z);dt+=b*a*np.cos(a*z)
        g=np.log(abs(th))-np.pi*z.imag**2/self.A
        d=dt/th
        return g,np.stack((d.real,-d.imag-2*np.pi*z.imag/self.A),axis=-1)

    def invwp(self,x,y):
        best=None
        for i in range(1,9):
            for j in range(1,9):
                zz=(i/9+self.tau*j/9)
                sol=root(lambda v:cjson(self.wp(v[0]+1j*v[1])[0]-x),cjson(zz),tol=1e-11)
                z=sol.x[0]+1j*sol.x[1]
                xx,yy=self.wp(z)
                if abs(xx-x)<1e-7*(1+abs(x)):
                    if abs(yy-y)>abs(yy+y): z=-z
                    uv=self.uv(z)%1
                    z=self.z(uv)
                    err=abs(self.wp(z)[1]-y)/(1+abs(y))
                    if best is None or err<best[0]: best=(err,z)
                    if err<1e-10:return z
        if best is None or best[0]>1e-7: raise RuntimeError(('inverse wp failed',best))
        return best[1]


class Flex:
    def __init__(self,tau,s=None,kind='generic'):
        self.T=Torus(tau); T=self.T; g,h=T.g2,T.g3
        self.kind=kind; self.s=s
        if kind=='symmetric':
            self.p=np.array([0,(1+T.tau)/3,2*(1+T.tau)/3])
            self.den=np.array([1,0,0],complex)
            self.num=np.array([[0,0,1],[0,1,0],[1,0,0]],complex)
            self.metadata={'kind':kind}
        elif kind=='cover':
            c=np.sqrt(g/12); yd=np.sqrt(4*c**3-g*c-h)
            z=T.invwp(-2*c,yd)
            self.p=np.array([0,z,T.z(T.uv(-z)%1)])
            self.den=np.array([1,0,2*c],complex)
            self.num=np.array([[0,0,1],[0,1,0],[1,0,c]],complex)
            self.metadata={'kind':kind,'c':cjson(c)}
        else:
            s=complex(s); t=np.sqrt(g*s+3*h+g*g/(12*s)); k=np.sqrt(12*s)
            m=-k; b=-t+g/k
            xs=np.roots([4,-m*m,-g-2*m*b,-h-b*b]); ys=m*xs+b
            slopes=(12*xs*xs-g)/(2*ys)
            xd=slopes**2/4-2*xs
            yd=ys-slopes*(xs-xd)
            pts=np.stack((xd,yd,np.ones(3)),axis=1)
            den=np.cross(pts[0],pts[1]); den/=np.linalg.norm(den)
            denerr=np.max(np.abs(pts@den))/np.max(np.linalg.norm(pts,axis=1))
            self.den=den
            self.num=np.array([[1,0,-s],[0,1,-t],[-k,1,t+g/k]],complex)
            self.p=np.array([T.invwp(x,y) for x,y in zip(xd,yd)])
            self.metadata={'kind':kind,'s':cjson(s),'t':cjson(t),
              'critical_x':[cjson(z) for z in xs], 'critical_y':[cjson(z) for z in ys],
              'puncture_x':[cjson(z) for z in xd], 'puncture_y':[cjson(z) for z in yd],
              'satellite_collinearity_error':float(denerr),
              'hessian_residual':float(abs(12*s*s*g-12*s*t*t+36*s*h+g*g)/(1+abs(g*g)))}
        self.metadata.update(tau=cjson(T.tau),g2=cjson(g),g3=cjson(h),
            punctures=[cjson(z) for z in self.p],
            denominator=[cjson(z) for z in self.den],
            numerators=[[cjson(z) for z in row] for row in self.num])

    def differentials(self,z):
        x,y=self.T.wp(z)
        v=np.stack((x,y,np.ones_like(x)),axis=-1)
        return (v@self.num.T)/(v@self.den)[...,None]


# Dunavant degree-5 rule: weights sum to one.
BARY=np.array([[1/3,1/3,1/3],
 [0.059715871789770,0.470142064105115,0.470142064105115],
 [0.470142064105115,0.059715871789770,0.470142064105115],
 [0.470142064105115,0.470142064105115,0.059715871789770],
 [0.797426985353087,0.101286507323456,0.101286507323456],
 [0.101286507323456,0.797426985353087,0.101286507323456],
 [0.101286507323456,0.101286507323456,0.797426985353087]])
BW=np.array([.225]+[.132394152788506]*3+[.125939180544827]*3)


def quadrature_order(order):
    global BARY,BW
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    s,t=np.meshgrid(x,x,indexing='ij');ws,wt=np.meshgrid(w,w,indexing='ij')
    BARY=np.stack((1-s,s*(1-t),s*t),axis=-1).reshape(-1,3)
    BW=(2*s*ws*wt).ravel()


def mesh(F,N,rmin,ntheta,ratio=1.35):
    T=F.T; h=1/N
    uv=np.stack(np.meshgrid((np.arange(N)+.271)/N,(np.arange(N)+.319)/N,indexing='ij'),axis=-1).reshape(-1,2)
    z=T.z(uv)
    sep=np.array([min(abs(T.center(p-pp)) for pp in F.p if abs(p-pp)>1e-12) for p in F.p])
    for p,d in zip(F.p,sep):z=z[np.abs(T.center(z-p))>min(3.1*h,.24*d)]
    zlist=list(z); ring_ids=[]
    for p,d in zip(F.p,sep):
        radii=[rmin]
        while radii[-1]<min(3.8*h,.30*d):radii.append(radii[-1]*ratio)
        # A fixed angular count makes the cusp truncation reproducible.
        ids=[]
        for ir,r in enumerate(radii):
            # Rotated successive rings avoid cocircular quads whose numerical
            # Delaunay diagonal could differ between translated tile copies.
            theta=(np.arange(ntheta)+.17+.193*np.sin(.73*ir))*2*np.pi/ntheta
            zz=p+r*np.exp(1j*theta)
            if r==rmin:ids=list(range(len(zlist),len(zlist)+ntheta))
            zlist.extend(zz)
        ring_ids.append(ids)
    z=T.z(T.uv(np.array(zlist))%1); nv=len(z)
    tiles=[i+j*T.tau for i in (-1,0,1) for j in (-1,0,1)]
    zz=np.concatenate([z+off for off in tiles]); ids=np.tile(np.arange(nv),9)
    tri=Delaunay(np.stack((zz.real,zz.imag),axis=1)).simplices
    tz=zz[tri]; centers=np.mean(tz,axis=1); cuv=T.uv(centers)
    keep=(cuv[:,0]>=0)&(cuv[:,0]<1)&(cuv[:,1]>=0)&(cuv[:,1]<1)
    for p in F.p:keep &= np.abs(T.center(centers-p))>rmin*np.cos(np.pi/ntheta)*(1-1e-10)
    tri=tri[keep]; tz=zz[tri]; tri=ids[tri]
    dz1=tz[:,1]-tz[:,0]; dz2=tz[:,2]-tz[:,0]
    det=np.imag(np.conj(dz1)*dz2)
    neg=det<0; tri[neg]=tri[neg][:,[0,2,1]]; tz[neg]=tz[neg][:,[0,2,1]]
    det=np.abs(det); area=det/2
    grad=np.stack((np.stack(((tz[:,1]-tz[:,2]).imag,(tz[:,2]-tz[:,1]).real),axis=1),
                   np.stack(((tz[:,2]-tz[:,0]).imag,(tz[:,0]-tz[:,2]).real),axis=1),
                   np.stack(((tz[:,0]-tz[:,1]).imag,(tz[:,1]-tz[:,0]).real),axis=1)),axis=1)/det[:,None,None]
    stiff=area[:,None,None]*np.einsum('tix,tjx->tij',grad,grad)
    rows=np.repeat(tri,3,axis=1).ravel(); cols=np.tile(tri,(1,3)).ravel()
    K=coo_matrix((stiff.ravel(),(rows,cols)),shape=(nv,nv)).tocsr()
    # Quadrature for the exact radial Robin flux on the polygon boundary.
    gx,gw=np.polynomial.legendre.leggauss(8); gx=(gx+1)/2;gw=gw/2
    eb=np.stack((1-gx,gx),axis=1)
    be=[]; bz=[]; bp=[]
    for p,ii in zip(F.p,ring_ids):
        for k in range(ntheta):
            i,j=ii[k],ii[(k+1)%ntheta]
            a=p+T.center(z[i]-p);b=p+T.center(z[j]-p)
            be.append([i,j]);bz.append([a,b]);bp.append(p)
    be=np.array(be);bz=np.array(bz);bp=np.array(bp)
    bq=bz@eb.T; rr=abs(bq-bp[:,None]);length=abs(bz[:,1]-bz[:,0])
    # Polygon edge normal has radial projection -cos(pi/ntheta)*rmin/r.
    radproj=rmin*np.cos(np.pi/ntheta)/rr
    bweight=length[:,None]*gw[None,:]*radproj
    rhs=np.bincount(be.ravel(),weights=np.einsum('eq,qi->ei',bweight/rr,eb).ravel(),minlength=nv)
    br=np.repeat(be,2,axis=1).ravel();bc=np.tile(be,(1,2)).ravel()
    # Every retained edge must be paired or belong to an intended cusp boundary.
    edges=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
    ee,count=np.unique(edges,axis=0,return_counts=True)
    boundary=ee[count==1]
    observed=set(map(tuple,boundary));expected=set(map(tuple,np.sort(be,axis=1)))
    if observed!=expected:
        extra=list(observed-expected);missing=list(expected-observed)
        raise AssertionError({'extra_edges':extra[:10],'missing_edges':missing[:10],
          'extra_positions':[[cjson(z[i]),cjson(z[j])] for i,j in extra[:10]],
          'counts':[len(extra),len(missing)]})
    assert np.max(count)<=2
    return dict(z=z,tri=tri,tz=tz,area=area,grad=grad,K=K,rows=rows,cols=cols,
      be=be,eb=eb,bweight=bweight,rhs=rhs,br=br,bc=bc,nv=nv,
      bq=bq,sep=sep,ring_ids=ring_ids,rmin=rmin,ntheta=ntheta,
      boundary_flux=float(np.sum(rhs)),flat_area=float(np.sum(area)))


def cusp_background(F,M,kind='green'):
    def evaluate(z):
        val=np.zeros(z.shape); grad=np.zeros(z.shape+(2,))
        for p,d in zip(F.p,M['sep']):
            if kind=='green':
                gg,ggrad=F.T.green(z-p);val-=gg;grad-=ggrad
                continue
            dz=F.T.center(z-p);r=abs(dz);R=.23*d
            t=np.clip((r-R/2)/(R/2),0,1)
            chi=1-10*t**3+15*t**4-6*t**5
            chip=(-30*t**2+60*t**3-30*t**4)/(R/2)
            L=1-np.log(r);f=-np.log(r)-np.log(L);fp=(-1+1/L)/r
            val+=chi*f
            der=chip*f+chi*fp
            grad+=der[...,None]*np.stack((dz.real,dz.imag),axis=-1)/r[...,None]
        return val,grad
    M['u0nodes']=evaluate(M['z'])[0]
    M['u0q'],gq=evaluate(M['tz']@BARY.T)
    M['u0b']=evaluate(M['bq'])[0]
    if 'gradq' in M:
        local=np.einsum('tqx,tqix,q,t->ti',gq,M['gradq'],BW,M['area'])
    else:
        gint=np.einsum('tqx,q,t->tx',gq,BW,M['area'])
        local=np.einsum('tx,tix->ti',gint,M['grad'])
    M['load0']=np.bincount(M['tri'].ravel(),weights=local.ravel(),minlength=M['nv'])


def quadratic_elements(F,M):
    old=M['tri'];nv=M['nv'];nt=len(old)
    ed=np.sort(np.concatenate((old[:,[0,1]],old[:,[1,2]],old[:,[2,0]])),axis=1)
    edges,inv=np.unique(ed,axis=0,return_inverse=True)
    mid=nv+inv.reshape(3,nt).T
    zmid=M['z'][edges[:,0]]+F.T.center(M['z'][edges[:,1]]-M['z'][edges[:,0]])/2
    M['z']=np.r_[M['z'],F.T.z(F.T.uv(zmid)%1)]
    M['tri']=np.concatenate((old,mid),axis=1);M['nv']=len(M['z'])
    l=BARY;S=np.concatenate((l*(2*l-1),4*l[:,[0,1,2]]*l[:,[1,2,0]]),axis=1)
    gl=M['grad'];gg=[]
    for i in range(3):gg.append((4*l[:,i]-1)[None,:,None]*gl[:,None,i,:])
    for i,j in [(0,1),(1,2),(2,0)]:gg.append(4*(l[None,:,i,None]*gl[:,None,j,:]+l[None,:,j,None]*gl[:,None,i,:]))
    gq=np.stack(gg,axis=2);M['gradq']=gq;M['shape']=S
    stiff=np.einsum('tqix,tqjx,q,t->tij',gq,gq,BW,M['area'])
    tri=M['tri'];M['rows']=np.repeat(tri,6,axis=1).ravel();M['cols']=np.tile(tri,(1,6)).ravel()
    M['K']=coo_matrix((stiff.ravel(),(M['rows'],M['cols'])),shape=(M['nv'],M['nv'])).tocsr()
    lookup={tuple(e):nv+i for i,e in enumerate(edges)}
    be=M['be'];M['be']=np.c_[be,[lookup[tuple(e)] for e in np.sort(be,axis=1)]]
    t=M['eb'][:,1];M['eb']=np.stack(((1-t)*(1-2*t),t*(2*t-1),4*t*(1-t)),axis=1)
    M['rhs']=np.r_[M['rhs'],np.zeros(len(edges))]
    # Recompute the inhomogeneous Robin load with the quadratic edge basis.
    pcoords=[]
    for row in M['be']:
        zz=M['z'][row[0]]
        pcoords.append(min(F.p,key=lambda p:abs(F.T.center(zz-p))))
    pp=np.array(pcoords)
    rr=np.stack([abs(F.T.center(M['bq'][i]-p)) for i,p in enumerate(pp)])
    M['rhs']=np.bincount(M['be'].ravel(),weights=((M['bweight']/rr)@M['eb']).ravel(),minlength=M['nv'])
    M['br']=np.repeat(M['be'],3,axis=1).ravel();M['bc']=np.tile(M['be'],(1,3)).ravel()


def nonlinear(M,u,hessian=True):
    tri=M['tri'];be=M['be'];nv=M['nv']
    basis=M.get('shape',BARY)
    uq=u[tri]@basis.T+M.get('u0q',0); expo=np.exp(2*uq)*M['area'][:,None]*BW
    inverse=np.exp(-2*uq)*M.get('qabs2',0)*M['area'][:,None]*BW
    ub=u[be]@M['eb'].T+M.get('u0b',0); exb=np.exp(ub)*M['bweight']
    vec=np.bincount(tri.ravel(),weights=((expo-inverse)@basis).ravel(),minlength=nv)
    vec+=np.bincount(be.ravel(),weights=(exb@M['eb']).ravel(),minlength=nv)
    res=M['K']@u+M.get('load0',0)+vec-M['rhs']
    energy=.5*u@(M['K']@u)+np.sum(u*M.get('load0',0))+.5*np.sum(expo+inverse)+np.sum(exb)-M['rhs']@u
    if not hessian:return energy,res,float(np.sum(expo)),float(np.sum(exb))
    hlocal=2*np.einsum('tq,qi,qj->tij',expo+inverse,basis,basis)
    hb=np.einsum('eq,qi,qj->eij',exb,M['eb'],M['eb'])
    H=M['K']+coo_matrix((hlocal.ravel(),(M['rows'],M['cols'])),shape=(nv,nv)).tocsr()
    H+=coo_matrix((hb.ravel(),(M['br'],M['bc'])),shape=(nv,nv)).tocsr()
    return energy,res,H,float(np.sum(expo)),float(np.sum(exb))


def cusp_gram_tail(F,M,u,scales):
    R=M['rmin'];nt=M['ntheta'];gx,gw=np.polynomial.legendre.leggauss(12)
    angles=gx*np.pi/nt;weights=gw*np.pi/nt*nt
    rad=R*np.cos(np.pi/nt)/np.cos(angles)
    G=np.zeros((3,3),complex);constants=[]
    uu=u+M.get('u0nodes',0)
    for p,ii in zip(F.p,M['ring_ids']):
        C=np.mean(np.exp(-uu[ii])/R)+np.log(R)
        L=C-np.log(rad)
        integral=np.sum(weights*.5*rad**2*(L**2+L+.5))
        delta=R*np.exp(2j*np.pi*(np.arange(64)+.123)/64)
        residue=np.mean(delta[:,None]*F.differentials(p+delta),axis=0)/scales
        G+=integral*np.outer(residue,np.conj(residue));constants.append(float(C))
    return G,constants


def solve(F,N=32,rmin=1e-3,ntheta=32,ratio=1.35,save=None,background=False,degree=1):
    start=time.time();M=mesh(F,N,rmin,ntheta,ratio)
    if degree==2:quadratic_elements(F,M)
    if background:cusp_background(F,M,kind=background if isinstance(background,str) else 'green')
    distance=np.min(np.stack([abs(F.T.center(M['z']-p)) for p in F.p]),axis=0)
    u=-np.log(distance)-np.log(np.maximum(.8-np.log(distance),.1))+.4-M.get('u0nodes',0)
    history=[]
    for it in range(30):
        E,r,H,area,tail=nonlinear(M,u)
        history.append([float(E),float(np.max(abs(r)))])
        if np.max(abs(r))<2e-10:break
        du=spsolve(H,-r);step=1.
        while step>1e-7:
            ne,nr,_,_=nonlinear(M,u+step*du,False)
            if ne<=E+1e-4*step*(r@du)+1e-11:break
            step*=.5
        u+=step*du
    E,r,area,tail=nonlinear(M,u,False)
    zq=M['tz']@BARY.T
    q=F.differentials(zq)
    # Normalize each differential first, then form its Hermitian WP Gram matrix.
    scales=np.sqrt(np.mean(np.abs(q)**2,axis=(0,1)))
    q=q/scales
    weight=M['area'][:,None]*BW*np.exp(-2*(u[M['tri']]@M.get('shape',BARY).T+M.get('u0q',0)))
    G=np.einsum('tqi,tqj,tq->ij',q,np.conj(q),weight)
    Graw=G.copy();tailG,cuspC=cusp_gram_tail(F,M,u,scales);G+=tailG
    v=G[:2,2];P=float(np.real(np.conj(v)@np.linalg.solve(G[:2,:2],v)/G[2,2]))
    # Since G convention is <f_i,f_j>=f_i conjugate(f_j), projection uses
    # v^* G^{-1} v (equivalently transpose coefficients for linear-first convention).
    correlations=G[:2,2]/np.sqrt(G.diagonal()[:2]*G[2,2])
    answer={'geometry':F.metadata,'N':N,'rmin':rmin,'ntheta':ntheta,'ratio':ratio,'background':background,'degree':degree,
      'vertices':M['nv'],'triangles':len(M['tri']), 'iterations':it,
      'newton_residual':float(np.max(abs(r))),'hyperbolic_area_truncated':area,
      'cusp_tail_area_robin':tail,'gauss_bonnet_error':float(area+tail-6*np.pi),
      'flat_area':M['flat_area'],'boundary_flux':M['boundary_flux'],
      'projection_squared':P,'correlations':[cjson(c) for c in correlations],
      'gram':[[cjson(c) for c in row] for row in G],
      'gram_truncated':[[cjson(c) for c in row] for row in Graw],
      'gram_cusp_tail':[[cjson(c) for c in row] for row in tailG],
      'cusp_constants':cuspC,'tail_correction':True,
      'scales':scales.tolist(),'quadrature_points':len(BW),'seconds':time.time()-start,'history':history}
    if save:
        Path(save).with_suffix('.json').write_text(json.dumps(answer,indent=2))
        np.savez_compressed(Path(save).with_suffix('.npz'),u=u+M.get('u0nodes',0),v=u,z=M['z'],tri=M['tri'],tz=M['tz'])
    return answer


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tau',nargs=2,type=float,default=[.23,1.08]);p.add_argument('--s',nargs=2,type=float,default=[5,2]);p.add_argument('--kind',default='generic');p.add_argument('--N',type=int,default=32);p.add_argument('--rmin',type=float,default=1e-3);p.add_argument('--ntheta',type=int,default=32);p.add_argument('--ratio',type=float,default=1.35);p.add_argument('--save');args=p.parse_args()
    F=Flex(complex(*args.tau),complex(*args.s),args.kind)
    ans=solve(F,args.N,args.rmin,args.ntheta,args.ratio,args.save)
    print(json.dumps({k:v for k,v in ans.items() if k not in ['gram','history']},indent=2),flush=True)
