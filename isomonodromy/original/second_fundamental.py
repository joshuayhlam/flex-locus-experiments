"""Unconstrained ambient WP connection and full II of algebraic flex surfaces.

The metric solver never receives the flex equation or a proposed orthogonality.
Complex metric convention: norm(v)^2 = v.conj().T @ g @ v.
"""
import argparse, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
import flex_wp as m
from scipy.sparse import coo_matrix
ORIGINAL_MESH=m.mesh


def cj(a):
    a=np.asarray(a)
    return np.stack((a.real,a.imag),axis=-1).tolist()


def uncj(a):
    a=np.asarray(a); return a[...,0]+1j*a[...,1]


class Ambient:
    def __init__(self,x,offset=0j):
        self.x=np.array(x,complex); self.T=m.Torus(x[0])
        self.offset=offset;raw=offset+np.r_[0j,self.x[1:]]
        self.lattice_shifts=np.floor(self.T.uv(raw))
        self.p=self.T.z(self.T.uv(raw)%1)
        self.xy=[self.T.wp(w) for w in self.x[1:]]
        self.metadata={'ambient_coordinates':cj(x),'punctures':cj(self.p)}

    def differentials(self,z):
        X,Y=self.T.wp(z-self.offset)
        return np.stack([np.ones_like(X)]+[(Y+y)/(2*(X-x)) for x,y in self.xy],axis=-1)

    def duality(self):
        # The a-cycle lies below all three chosen puncture lifts. Stokes gives
        # <q,d/dtau>=i/2 int_a q/dz, <q,d/dw_j>=-pi Res_j(q/dz).
        w=self.x[1:]; z=self.T.center(w)
        th=np.zeros_like(z);dt=np.zeros_like(z)
        for n in range(13):
            a=(2*n+1)*np.pi;b=2*(-1)**n*np.exp(1j*np.pi*self.T.tau*(n+.5)**2)
            th+=b*np.sin(a*z);dt+=b*a*np.cos(a*z)
        shift=np.rint(self.T.uv(w-z)[:,1])
        periods=dt/th-2j*np.pi*shift
        J=np.diag([.5j,-np.pi,-np.pi]).astype(complex)
        J[1:,0]=.5j*periods
        return J


class FlexChart:
    def __init__(self,tau,s):
        self.u=np.array([tau,s],complex); self.F=m.Flex(tau,s)
        self.ref=self.F.p.copy()
        self.shifts=np.floor(self.F.T.uv(self.ref[1:]-self.ref[0]))
        self.x=self.at(self.u)

    def at(self,u):
        F=m.Flex(*u); T=F.T
        perm=min(itertools.permutations(range(3)),key=lambda pp:sum(abs(T.center(F.p[j]-r))**2 for j,r in zip(pp,self.ref)))
        p=np.array([r+T.center(F.p[j]-r) for j,r in zip(perm,self.ref)])
        w=p[1:]-p[0]-self.shifts[:,0]-u[0]*self.shifts[:,1]
        return np.r_[u[0],w]

    def jets(self,scale=1,points=32):
        radii=np.array([.015,.15])*scale
        phases=np.exp(2j*np.pi*np.arange(points)/points)
        def directional(d):
            d=np.asarray(d)
            vals=np.array([self.at(self.u+t*d)-self.x for t in phases])
            return np.mean(vals/phases[:,None],axis=0),2*np.mean(vals/phases[:,None]**2,axis=0)
        a,aa=directional([radii[0],0]); b,bb=directional([0,radii[1]])
        _,cc=directional(radii)
        D=np.stack([a/radii[0],b/radii[1]],axis=1)
        DD=np.zeros((3,2,2),complex)
        DD[:,0,0]=aa/radii[0]**2; DD[:,1,1]=bb/radii[1]**2
        DD[:,0,1]=DD[:,1,0]=(cc-aa-bb)/(2*np.prod(radii))
        return D,DD


class Metric:
    def __init__(self,out,N=32,rmin=1e-4,ntheta=64,ratio=1.2,quad=4,common=False,shift=False):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.cfg=dict(N=N,rmin=rmin,ntheta=ntheta,ratio=ratio,degree=2,background='green')
        self.quad=quad;m.quadrature_order(quad)
        self.common=common;self.anchor=None;self.template=None
        self.shift=shift;self.offset=.173+.237j if shift else 0j

    def __call__(self,x):
        payload_dict=dict(x=cj(x),quad=self.quad,anchor=None if self.anchor is None else cj(self.anchor),**self.cfg)
        if self.shift:payload_dict['offset']=cj(self.offset)
        payload=json.dumps(payload_dict,sort_keys=True)
        key=hashlib.sha256(payload.encode()).hexdigest()[:20]
        path=self.out/(key+'.json')
        A=Ambient(x,self.offset)
        if path.exists(): ans=json.loads(path.read_text())
        else:
            ans=m.solve(A,**self.cfg)
            path.write_text(json.dumps(ans,indent=2))
            print(json.dumps({'metric':key,'N':self.cfg['N'],'seconds':round(ans['seconds'],2),'newton':ans['newton_residual']}),flush=True)
        scales=np.array(ans['scales']);G=uncj(ans['gram'])*np.outer(scales,scales)
        J=A.duality();g=J.conj().T@np.linalg.solve(G,J)
        return g

    def connection(self,x,h):
        if self.common:
            self.anchor=np.array(x);A=Ambient(x,self.offset)
            self.template=ORIGINAL_MESH(A,self.cfg['N'],self.cfg['rmin'],self.cfg['ntheta'],self.cfg['ratio'])
            m.mesh=self.moving_mesh
        else:m.mesh=ORIGINAL_MESH
        g=self(x);dg=[];dxs=[];dys=[]
        for k in range(3):
            d=np.zeros(3,complex);d[k]=h
            dx=(self(x+d)-self(x-d))/(2*h)
            dy=(self(x+1j*d)-self(x-1j*d))/(2*h)
            dxs.append(dx);dys.append(dy)
            dg.append((dx-1j*dy)/2)
        C=np.array([np.linalg.solve(g,v) for v in dg]) # C[i,k,j]=Gamma^k_ij
        self.real_connection=levi_civita(g,np.array(dxs+dys))
        return g,C

    def moving_mesh(self,F,*unused):
        M=self.template.copy();A=Ambient(self.anchor,self.offset);T=A.T
        dt=F.x[0]-self.anchor[0];dp=np.r_[0j,F.x[1:]-self.anchor[1:]]
        dp-=dt*A.lattice_shifts[:,1]
        def warp(z):
            uv=T.uv(z);ans=z+dt*uv[...,1]
            for j,p in enumerate(A.p):
                dz=T.center(z-p);R=min(.12*M['sep'][j],.10)
                v=np.clip((abs(dz)-R)/R,0,1)
                chi=1-10*v**3+15*v**4-6*v**5
                ans+=chi*(dp[j]-dt*(T.uv(p)[1]+dz.imag/T.A))
            return ans
        M['z']=warp(M['z']);tz=warp(M['tz']);M['tz']=tz
        det=np.imag(np.conj(tz[:,1]-tz[:,0])*(tz[:,2]-tz[:,0]))
        if det.min()<=0:raise RuntimeError('Moving mesh inverted a triangle')
        area=det/2
        grad=np.stack((np.stack(((tz[:,1]-tz[:,2]).imag,(tz[:,2]-tz[:,1]).real),axis=1),
                       np.stack(((tz[:,2]-tz[:,0]).imag,(tz[:,0]-tz[:,2]).real),axis=1),
                       np.stack(((tz[:,0]-tz[:,1]).imag,(tz[:,1]-tz[:,0]).real),axis=1)),axis=1)/det[:,None,None]
        stiff=area[:,None,None]*np.einsum('tix,tjx->tij',grad,grad)
        M.update(area=area,grad=grad,K=coo_matrix((stiff.ravel(),(M['rows'],M['cols'])),shape=(M['nv'],M['nv'])).tocsr(),
                 bq=warp(M['bq']),flat_area=float(area.sum()))
        return M


def real_metric(g):
    return np.block([[g.real,-g.imag],[g.imag,g.real]])


def levi_civita(g,dg):
    R=real_metric(g);dR=np.array([real_metric(d) for d in dg])
    # A[l,i,j] is twice the lowered Christoffel symbol.
    A=np.empty((6,6,6))
    for l in range(6):
        for i in range(6):
            for j in range(6):A[l,i,j]=dR[i,l,j]+dR[j,l,i]-dR[l,i,j]
    return .5*np.linalg.solve(R,A.reshape(6,36)).reshape(6,6,6)


def real_second_form(g,LC,D,DD,E,normal):
    U=np.c_[E,1j*E]; Z=D@U;Z=np.r_[Z.real,Z.imag]
    Q=np.einsum('kab,ac,bd->kcd',DD,U,U); Q=np.r_[Q.real,Q.imag]
    W=Q+np.einsum('kij,ia,jb->kab',LC,Z,Z)
    normals=np.stack([normal,1j*normal],axis=1);normals=np.r_[normals.real,normals.imag]
    B=np.einsum('kn,kl,lab->nab',normals,real_metric(g),W)
    return dict(B_real=B.tolist(),epsilon_II_real_half=float(np.linalg.norm(B)/2))


def second_form(g,C,D,DD):
    induced=D.conj().T@g@D
    R=np.linalg.cholesky(induced).conj().T
    E=np.linalg.inv(R)  # D E has WP-orthonormal columns
    c=np.cross(D[:,0],D[:,1])
    norm=np.sqrt(np.real(c@np.linalg.solve(g,c.conj())))
    c=c/norm
    normal=np.linalg.solve(g,c.conj())
    acceleration=DD+np.einsum('ikj,ia,jb->kab',C,D,D)
    B=np.einsum('k,kab->ab',c,acceleration)
    Borth=E.T@B@E
    # Keep antisymmetry as an independent Kähler/derivative error diagnostic.
    sym=(Borth+Borth.T)/2
    torsion=C-C.transpose(2,1,0)
    return dict(induced=cj(induced),orthonormal_parameter_frame=cj(E),
        unit_normal=cj(normal),unit_conormal=cj(c),B=cj(B),B_orthonormal=cj(Borth),
        epsilon_II=float(np.linalg.norm(sym)),II_antisymmetry=float(np.linalg.norm(Borth-Borth.T)/2),
        connection_torsion_raw=float(np.linalg.norm(torsion)))


def test(args):
    chart=FlexChart(complex(*args.tau),complex(*args.s)); D,DD=chart.jets()
    D2,DD2=chart.jets(scale=.7,points=40)
    metric=Metric(Path(args.out)/'metrics',args.N,args.rmin,args.ntheta,args.ratio,args.quad,args.common,args.shift)
    g,C=metric.connection(chart.x,args.h)
    ans=second_form(g,C,D,DD)
    # The control is an actual holomorphic immersion germ, with identical
    # point and first jet, and prescribed Hessian in the unit normal.
    E=uncj(ans['orthonormal_parameter_frame']); normal=uncj(ans['unit_normal'])
    ans.update(real_second_form(g,metric.real_connection,D,DD,E,normal))
    target=np.array([[.12,.07+.03j],[.07+.03j,-.09]],complex)
    bend=np.linalg.inv(E).T@target@np.linalg.inv(E)
    control=second_form(g,C,D,DD+np.einsum('k,ab->kab',normal,bend))
    control.update(real_second_form(g,metric.real_connection,D,DD+np.einsum('k,ab->kab',normal,bend),E,normal))
    ans.update(control=control,control_prescribed_B_increment=cj(target),
        control_recovery_error=float(np.linalg.norm(uncj(control['B_orthonormal'])-uncj(ans['B_orthonormal'])-target)),
        x=cj(chart.x),tau=cj(chart.u[0]),s=cj(chart.u[1]),D=cj(D),DD=cj(DD),
        g=cj(g),connection=cj(C),real_connection=metric.real_connection.tolist(),geometry_first_derivative_check=float(np.max(abs(D-D2))),
        geometry_second_derivative_check=float(np.max(abs(DD-DD2))),h=args.h,
        metric_config=dict(quad=args.quad,common=args.common,shift=args.shift,**metric.cfg))
    Path(args.out).mkdir(parents=True,exist_ok=True)
    name=args.label or f'N{args.N}-h{args.h:g}'
    (Path(args.out)/(name+'.json')).write_text(json.dumps(ans,indent=2))
    print('RESULT',json.dumps({k:ans[k] for k in ['epsilon_II','II_antisymmetry','connection_torsion_raw','B_orthonormal','geometry_second_derivative_check','control_recovery_error']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tau',nargs=2,type=float,default=[.23,1.08]);p.add_argument('--s',nargs=2,type=float,default=[5,2])
    p.add_argument('--N',type=int,default=32);p.add_argument('--h',type=float,default=.002)
    p.add_argument('--rmin',type=float,default=1e-4);p.add_argument('--ntheta',type=int,default=64)
    p.add_argument('--ratio',type=float,default=1.2);p.add_argument('--quad',type=int,default=4)
    p.add_argument('--out',default='run');p.add_argument('--label')
    p.add_argument('--common',action='store_true')
    p.add_argument('--shift',action='store_true',help='Translate the mesh fundamental domain away from a puncture')
    test(p.parse_args())
