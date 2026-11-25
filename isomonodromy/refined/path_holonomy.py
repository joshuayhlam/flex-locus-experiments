"""Unprojected parallel transport along geometrically marked loops.

Each polyline segment is split exactly at finite-element cell boundaries.
Fourth-order commutator-free Magnus integration is used within each cell.
Neither flatness nor a character equation is imposed on the measured matrices.
"""
import numpy as np
from scipy.spatial import cKDTree,Delaunay
from seed_higgs import P,adj,exp_hermitian,background_gradient,cutoff
from connection_check import exp2,inv,I

class Field:
    def __init__(self,o):
        self.o=o;self.M=M=o.M;self.F=o.F;self.T=T=o.F.T
        centers=M['tz'].mean(axis=1);self.centers=T.z(T.uv(centers)%1)
        tiles=np.array([a+b*T.tau for a in (-1,0,1) for b in (-1,0,1)])
        cc=(self.centers[None,:]+tiles[:,None]).ravel()
        self.tree=cKDTree(np.c_[cc.real,cc.imag]);self.nt=len(centers)
        nv=int(M['tri'][:,:3].max())+1;zz=(M['z'][:nv][None,:]+tiles[:,None]).ravel()
        self.delaunay=Delaunay(np.c_[zz.real,zz.imag])
        lookup={tuple(sorted(row)):i for i,row in enumerate(M['tri'][:,:3])}
        self.parent=np.array([lookup.get(tuple(sorted(row%nv)),-1) for row in self.delaunay.simplices])

    def bary(self,z,pa):
        v=self.T.center(z-self.M['tz'][pa,0]);vv=np.stack((v.real,v.imag),axis=-1)
        l=np.einsum('...ix,...x->...i',self.M['grad'][pa],vv);l[...,0]+=1
        return l

    def locate(self,z,k=24):
        zz=self.T.z(self.T.uv(z)%1)
        ix=self.delaunay.find_simplex(np.c_[zz.real,zz.imag]);pa=self.parent[ix]
        if np.any(ix<0)|np.any(pa<0):raise ValueError('path point lies outside retained mesh')
        return pa

    def split_path(self,path,max_length=.007,neighbors=32):
        # Find each containing cell using the original periodic Delaunay
        # triangulation, not a nearest-centroid search (which fails in the
        # strongly graded cusp-to-bulk transition).
        a=np.asarray(path[:-1]);b=np.asarray(path[1:]);da=b-a
        row=np.arange(len(a));lo=np.zeros(len(a));hi=np.ones(len(a));parts=[];snapped=0.
        for iteration in range(100):
            if not len(row):break
            mid=a[row]+.5*(lo+hi)*da[row];pa=self.locate(mid)
            tiny=(hi-lo)*abs(da[row])<1e-10
            if np.any(tiny):
                snapped+=float(np.sum((hi[tiny]-lo[tiny])*abs(da[row[tiny]])))
                parts.append((row[tiny],lo[tiny],hi[tiny],pa[tiny]))
                row,lo,hi,pa=[x[~tiny] for x in [row,lo,hi,pa]]
                if not len(row):break
            start=a[row]+lo*da[row];length=(hi-lo)*da[row]
            la=self.bary(start,pa);dl=np.einsum('nix,nx->ni',self.M['grad'][pa],np.c_[length.real,length.imag])
            with np.errstate(divide='ignore',invalid='ignore'):tt=-la/dl
            left=np.clip(np.max(np.where(dl>1e-14,tt,-np.inf),axis=1),0,1)
            right=np.clip(np.min(np.where(dl<-1e-14,tt,np.inf),axis=1),0,1)
            l=lo+left*(hi-lo);r=lo+right*(hi-lo)
            parts.append((row.copy(),l,r,pa))
            takeleft=(l-lo)>1e-10;takeright=(hi-r)>1e-10
            nr=np.r_[row[takeleft],row[takeright]];nl=np.r_[lo[takeleft],r[takeright]];nh=np.r_[l[takeleft],hi[takeright]]
            row,lo,hi=nr,nl,nh
        if len(row):raise ValueError(('cell splitting did not terminate',float(np.max((hi-lo)*abs(da[row])))))
        if snapped>1e-8:raise ValueError(('excessive intersection roundoff',snapped))
        self.last_split_snap_length=snapped
        row,lo,hi,pa=[np.concatenate([p[j] for p in parts]) for j in range(4)]
        order=np.lexsort((lo,row));row,lo,hi,pa=[q[order] for q in [row,lo,hi,pa]]
        coverage=np.bincount(row,weights=hi-lo,minlength=len(a))
        if np.max(abs(coverage-1))>1e-7:raise ValueError('incomplete cell splitting')
        return a[row]+lo*da[row],a[row]+hi*da[row],pa

    def connection(self,z,pa):
        o=self.o;M=self.M;l=self.bary(z,pa);g=M['grad'][pa]
        shape=np.concatenate((l*(2*l-1),4*l[:,[0,1,2]]*l[:,[1,2,0]]),axis=1)
        grad=[(4*l[:,i,None]-1)*g[:,i] for i in range(3)]
        for i,j in [(0,1),(1,2),(2,0)]:grad.append(4*(l[:,i,None]*g[:,j]+l[:,j,None]*g[:,i]))
        grad=np.stack(grad,axis=1);nodes=M['tri'][pa]
        u=np.einsum('ni,ni->n',o.u[nodes],shape)
        ug=np.einsum('ni,nix->nx',o.u[nodes],grad)
        s=np.einsum('nia,ni->na',o.s[nodes],shape)
        sg=np.einsum('nia,nix->nxa',o.s[nodes],grad)
        for p in self.F.p:u-=self.T.green(z-p)[0]
        gg=ug+background_gradient(self.F,z);uz=.5*(gg[:,0]-1j*gg[:,1])
        psi=self.F.differentials(z);Q=psi@o.qcoeff
        phi=np.zeros((len(z),2,2),complex);phi[:,0,1]=.5*Q*np.exp(-u);phi[:,1,0]=.5*np.exp(u)
        chi,_=cutoff(self.F,z,M['sep'])
        f=o.amplitude*o.coefficients['f_scaling']*chi*(psi.conj()@o.fcoeff)
        if o.control:f+=o.amplitude*o.coefficients['f_scaling']*chi*.35*o.coefficients['norms'][0]/o.coefficients['norms'][2]*psi[:,2].conj()
        bc=.5*uz.conj()[:,None,None]*P[2]+f[:,None,None]*phi
        H,Hi,Hz,Hb=exp_hermitian(s,sg[:,0,:],sg[:,1,:])
        Az=Hi@Hz-Hi@adj(bc)@H+phi;Ab=bc+Hi@adj(phi)@H
        return Az,Ab

    def holonomy(self,path,subdivide=1):
        a,b,pa=self.split_path(path)
        if subdivide>1:
            f=np.arange(subdivide)/subdivide;dz=(b-a)
            aa=(a[:,None]+dz[:,None]*f).ravel();bb=aa+np.repeat(dz/subdivide,subdivide)
            a,b,pa=aa,bb,np.repeat(pa,subdivide)
        dz=b-a;c1=.5-np.sqrt(3)/6;c2=.5+np.sqrt(3)/6
        az1,ab1=self.connection(a+c1*dz,pa);az2,ab2=self.connection(a+c2*dz,pa)
        A1=az1*dz[:,None,None]+ab1*dz.conj()[:,None,None]
        A2=az2*dz[:,None,None]+ab2*dz.conj()[:,None,None]
        alpha=(3-2*np.sqrt(3))/12;beta=(3+2*np.sqrt(3))/12
        terms=exp2(beta*A1+alpha*A2)@exp2(alpha*A1+beta*A2)
        ans=I.copy()
        for term in terms:ans=ans@term
        return ans,dict(segments=len(a),determinant_error=float(abs(np.linalg.det(ans)-1)),intersection_roundoff_length=self.last_split_snap_length)

def evaluate_word(word,G):
    out=I.copy();names='tuvw'
    for c in word:
        g=G[names.index(c.lower())];out=out@(g if c.islower() else inv(g))
    return out

def residuals(G):
    tr=lambda word:np.trace(evaluate_word(word,G))
    terms=np.array([[tr('twuv'),tr('u')**2,-2],[tr('wuWUT'),-tr('uw')**2,2],[tr('uwv'),-tr('w')**2,2]],complex)
    f=terms.sum(axis=1)
    return f,abs(f)/np.sum(abs(terms),axis=1)

def character(G):
    words=['t','u','v','w','tu','tv','tw','uv','uw','vw','tuv','tuw','tvw','uvw','tuvw']
    return np.array([np.trace(evaluate_word(w,G)) for w in words])

def parabolic_projection(G):
    """Correct only prescribed cusp traces; the Betti polynomials are unused."""
    original=G.copy();G=G/np.sqrt(np.linalg.det(G))[:,None,None]
    def r(g):return np.array([np.trace(evaluate_word(w,g))-v for w,v in zip(['t','uvw','tuwv'],[2,2,-2])])
    history=[]
    for iteration in range(20):
        rr=r(G);history.append(float(max(abs(rr))))
        if history[-1]<1e-10:break
        J=np.empty((3,12),complex);eps=1e-6
        for k in range(4):
            for a in range(3):
                plus=G.copy();minus=G.copy();plus[k]=G[k]@exp2(eps*P[a]);minus[k]=G[k]@exp2(-eps*P[a])
                J[:,3*k+a]=(r(plus)-r(minus))/(2*eps)
        delta=np.linalg.lstsq(J,-rr,rcond=1e-12)[0].reshape(4,3);step=1.
        while step>1e-7:
            Gn=G@exp2(step*np.einsum('na,aij->nij',delta,P))
            if np.linalg.norm(r(Gn))<np.linalg.norm(rr):break
            step*=.5
        if step<=1e-7:raise ValueError('parabolic correction stalled')
        G=Gn
    if max(abs(r(G)))>1e-8:raise ValueError('parabolic correction failed')
    return G,dict(cusp_residual_history=history,
        matrix_relative_change=float(np.linalg.norm(G-original)/np.linalg.norm(original)),
        character_relative_change=float(np.max(abs(character(G)-character(original))/(1+abs(character(G))))))
