"""Boundary horocycles compatible with the exactly parabolic monodromy."""
import numpy as np
from connection_check import I,P,inv,adj

def fit_boundary(data,F):
    edges=data['edges'];tri=data['tri'];lookup={tuple(e):k for k,e in enumerate(edges)}
    boundary={}
    for row in tri:
        for a,b in [(0,1),(1,2),(2,0)]:
            i,j=int(row[a]),int(row[b]);key=tuple(sorted((i,j)))
            if key in boundary:del boundary[key]
            else:boundary[key]=(i,j)
    nxt={i:j for i,j in boundary.values()};remaining=set(nxt);bp=np.zeros((len(data['z']),3));reports=[]
    while remaining:
        first=next(iter(remaining));i=first;indices=[];trans=[];U=I.copy()
        while True:
            indices.append(i);trans.append(U);remaining.remove(i);j=nxt[i]
            g=data['links'][lookup[tuple(sorted((i,j)))]];U=U@(g if i<j else inv(g));i=j
            if i==first:break
        indices=np.array(indices);trans=np.array(trans);C=U;N=-C-I
        if abs(np.trace(C)+2)>1e-8:raise RuntimeError('Cusp holonomy is not the prescribed parabolic')
        zz=data['z'][indices];p=min(F.p,key=lambda p:abs(F.T.center(zz[0]-p)))
        angle=np.unwrap(np.angle(F.T.center(zz-p)));x=(angle-angle[0])/(2*np.pi)
        # Boundary orientation is clockwise, while x counts positive traversal.
        x*=np.sign(angle[-1]-angle[0])
        E=I+x[:,None,None]*N;Ei=inv(E)
        candidates=Ei@trans@adj(trans)@adj(Ei)
        X=candidates.mean(axis=0);X/=np.sqrt(np.linalg.det(X)).real
        Xi=inv(trans)@E@X@adj(E)@adj(inv(trans))
        Xi/=np.sqrt(np.linalg.det(Xi)).real[:,None,None]
        pp=.5*np.einsum('aij,nji->na',P,Xi).real;bp[indices]=pp
        reports.append(dict(nodes=len(indices),radius=float(np.mean(abs(F.T.center(zz-p)))),
                            boundary_distance_max=float(np.arcsinh(np.linalg.norm(pp,axis=1)).max()),
                            nilpotent_error=float(np.linalg.norm(N@N)),trace_error=float(abs(np.trace(C)+2))))
    return bp,reports
