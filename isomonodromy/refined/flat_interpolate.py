"""Resample one flat bundle on another mesh without changing monodromy.

Each fine triangle has a local flat trivialization. Barycentric interpolation
chooses the smooth frame, while constant transition maps carry all holonomy.
This operation uses no Higgs, pencil, or spectral-line constraint.
"""
import numpy as np
from scipy.spatial import cKDTree
from connection_check import I,inv

class FlatInterpolator:
    def __init__(self,data,T):
        self.data=data;self.T=T;self.tri=tri=data['tri'];self.tz=tz=data['tz']
        edges=data['edges'];lookup={tuple(e):k for k,e in enumerate(edges)}
        def link(i,j):
            if i==j:return I
            g=data['links'][lookup[tuple(sorted((int(i),int(j))))]]
            return g if i<j else inv(g)
        self.local=np.array([[link(r[0],v) for v in r] for r in tri])
        d1=tz[:,1]-tz[:,0];d2=tz[:,2]-tz[:,0];det=(d1.conj()*d2).imag
        self.grad=np.stack((np.stack(((tz[:,1]-tz[:,2]).imag,(tz[:,2]-tz[:,1]).real),axis=1),
                   np.stack(((tz[:,2]-tz[:,0]).imag,(tz[:,0]-tz[:,2]).real),axis=1),
                   np.stack(((tz[:,0]-tz[:,1]).imag,(tz[:,1]-tz[:,0]).real),axis=1)),axis=1)/det[:,None,None]
        self.neighbor=np.full((len(tri),3),-1,int);table={}
        for t,r in enumerate(tri):
            for k in range(3):
                edge=tuple(sorted((int(r[(k+1)%3]),int(r[(k+2)%3]))))
                if edge in table:
                    u,l=table.pop(edge);self.neighbor[t,k]=u;self.neighbor[u,l]=t
                else:table[edge]=(t,k)
        shifts=np.array([i+j*T.tau for i in (-1,0,1) for j in (-1,0,1)])
        self.shifts=np.repeat(shifts,len(tri));self.ids=np.tile(np.arange(len(tri)),9)
        centers=np.tile(tz.mean(axis=1),9)+self.shifts
        self.tree=cKDTree(np.c_[centers.real,centers.imag])

    def bary(self,t,z,shift):
        v=z-self.tz[t,0]-shift
        l=self.grad[t]@np.array([v.real,v.imag]);l[0]+=1
        return l

    def locate(self,z):
        uv=self.T.uv(z);offset=self.T.z(np.floor(uv));zz=z-offset
        for n in (24,128,512):
            _,ii=self.tree.query([zz.real,zz.imag],k=n)
            ts=self.ids[ii];shifts=self.shifts[ii]+offset
            v=z-self.tz[ts,0]-shifts
            l=np.einsum('nix,nx->ni',self.grad[ts],np.c_[v.real,v.imag]);l[:,0]+=1
            m=np.min(l,axis=1);k=np.argmax(m)
            if m[k]>=-2e-8:return int(ts[k]),shifts[k]
        raise RuntimeError(f'Point outside fine domain: {z}, best bary {m[k]}')

    def B(self,t,z,shift):
        v=np.einsum('k,kij->ij',self.bary(t,z,shift),self.local[t])
        return v/np.sqrt(np.linalg.det(v))

    def transport(self,x,y):
        delta=self.T.center(y-x);y=x+delta
        t,shift=self.locate(x+1e-8*delta);start=self.B(t,x,shift);prod=I.copy();u=0.
        for it in range(2000):
            l=self.bary(t,x+u*delta,shift)
            dl=self.grad[t]@np.array([delta.real,delta.imag])
            remaining=np.where(dl<-1e-14,-l/np.minimum(dl,-1e-300),np.inf)
            remaining=np.where(remaining>=-1e-8,np.maximum(remaining,0),np.inf)
            k=int(np.argmin(remaining));end=u+remaining[k]
            if end>=1-1e-10:
                result=inv(start)@prod@self.B(t,y,shift)
                return result/np.sqrt(np.linalg.det(result)),it
            new=int(self.neighbor[t,k])
            if new<0:raise RuntimeError('Coarse edge enters fine cusp hole')
            shared=int(self.tri[t,(k+1)%3]);a=(k+1)%3;b=int(np.flatnonzero(self.tri[new]==shared)[0])
            trans=self.local[t,a]@inv(self.local[new,b]);prod=prod@trans
            shift=shift+self.tz[t,a]-self.tz[new,b]
            t=new;u=end
            # At a fine vertex, several edges can be crossed at the same time.
            # Choosing the face containing a tiny forward point avoids cycles.
            if remaining[k]<1e-10:
                probe=x+min(u+1e-8,1)*delta
                if np.min(self.bary(t,probe,shift))<-1e-8:
                    continue
        raise RuntimeError('Triangle walk failed to terminate')

    def resample(self,edges,z):
        # Shared edges, especially cusp-boundary edges, already have exact
        # transport data. Reuse these rather than walking along a boundary,
        # where roundoff can incorrectly select the excluded adjacent face.
        oldz=self.data['z'];nn=len(oldz)
        shifts=np.array([i+j*self.T.tau for i in (-1,0,1) for j in (-1,0,1)])
        copies=np.concatenate([oldz+s for s in shifts])
        tree=cKDTree(np.c_[copies.real,copies.imag])
        distance,node=tree.query(np.c_[z.real,z.imag]);node=node%nn
        shared=distance<1e-13
        oldedges={tuple(e):k for k,e in enumerate(self.data['edges'])}
        out=[];cross=[];reused=0
        for k,(i,j) in enumerate(edges):
            oi,oj=int(node[i]),int(node[j]);key=tuple(sorted((oi,oj)))
            if shared[i] and shared[j] and key in oldedges:
                g=self.data['links'][oldedges[key]]
                if oi>oj:g=inv(g)
                n=0;reused+=1
            else:
                try:g,n=self.transport(z[i],z[j])
                except RuntimeError as exc:
                    raise RuntimeError(f'Resampling edge {k}, nodes {(int(i),int(j))}, '
                                       f'points {(z[i],z[j])}: {exc}') from exc
            out.append(g);cross.append(n)
            if (k+1)%20000==0:print('resampled',k+1,'/',len(edges),flush=True)
        return np.array(out),dict(maximum_faces_crossed=max(cross),mean_faces_crossed=float(np.mean(cross)),
                                 reused_original_edges=reused)

def face_errors(data):
    lookup={tuple(e):k for k,e in enumerate(data['edges'])};tri=data['tri'];prod=np.broadcast_to(I,(len(tri),2,2)).copy()
    for a,b in [(0,1),(1,2),(2,0)]:
        ids=np.array([lookup[tuple(sorted((int(i),int(j))))] for i,j in tri[:,[a,b]]])
        g=data['links'][ids].copy();rev=tri[:,a]>tri[:,b];g[rev]=inv(g[rev]);prod=prod@g
    err=np.linalg.norm(prod-I,axis=(1,2))
    return dict(maximum=float(max(err)),median=float(np.median(err)))
