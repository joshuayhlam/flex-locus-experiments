"""Macroscopic contractible-loop checks for the approximate connection."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from connection_check import inv,I
from seed_higgs import cj

def audit(data,F):
    tri=data['tri'];centers=data['tz'].mean(axis=1)
    grid=(np.arange(17)+.5)/17;z=F.T.z(np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2))
    sep=np.min([abs(F.T.center(z-p)) for p in F.p],axis=0)
    center=z[np.argmax(sep)];lookup={tuple(e):k for k,e in enumerate(data['edges'])}
    ans=[]
    for radius in [.06,.10,.16]:
        selected=abs(F.T.center(centers-center))<radius
        boundary={}
        for row in tri[selected]:
            for a,b in [(0,1),(1,2),(2,0)]:
                i,j=int(row[a]),int(row[b]);key=tuple(sorted((i,j)))
                if key in boundary:del boundary[key]
                else:boundary[key]=(i,j)
        nxt={i:j for i,j in boundary.values()}
        if len(nxt)!=len(boundary) or not nxt:
            ans.append(dict(radius=radius,status='non_simple_mesh_boundary'));continue
        start=next(iter(nxt));i=start;product=I.copy();n=0
        while True:
            j=nxt[i];g=data['links'][lookup[tuple(sorted((i,j)))]]
            if i>j:g=inv(g)
            product=product@g;n+=1;i=j
            if i==start or n>len(nxt):break
        ans.append(dict(radius=radius,status='ok' if n==len(nxt) else 'multiple_components',edges=n,
                        holonomy_error=float(np.linalg.norm(product-I)),trace=cj(np.trace(product))))
    result=dict(center=cj(center),distance_to_nearest_puncture=float(max(sep)),loops=ans)
    if 'node_U' in data:result['noncontractible_loops']=trace_audit(data,F,center)
    return result

def trace_audit(data,F,center):
    edges=data['edges'];z=data['z'];i,j=edges.T
    dz=F.T.center(z[j]-z[i]);shift=np.rint(F.T.uv(dz-z[j]+z[i])).astype(int)
    weight=abs(dz)*np.exp(.5*(data['node_U'][i]+data['node_U'][j]))
    graph=coo_matrix((np.r_[weight,weight],(np.r_[i,j],np.r_[j,i])),shape=(len(z),len(z))).tocsr()
    base=int(np.argmin(abs(F.T.center(z-center))));dist,pred=dijkstra(graph,indices=base,return_predecessors=True)
    lookup={tuple(e):k for k,e in enumerate(edges)}
    frames=np.broadcast_to(I,(len(z),2,2)).copy();tile=np.zeros((len(z),2),int)
    for v in np.argsort(dist):
        if v==base:continue
        u=int(pred[v]);key=tuple(sorted((u,int(v))));k=lookup[key]
        sign=1 if u<v else -1;g=data['links'][k]
        if sign<0:g=inv(g)
        frames[v]=frames[u]@g;tile[v]=tile[u]+sign*shift[k]
    winding=tile[i]+shift-tile[j];ans=[]
    for a in [(1,0),(0,1)]:
        wanted=np.all(winding==a,axis=1)|np.all(winding==-np.array(a),axis=1)
        ids=np.flatnonzero(wanted)
        if not len(ids):continue
        k=int(ids[np.argmin((dist[i]+weight+dist[j])[ids])])
        hol=frames[i[k]]@data['links'][k]@inv(frames[j[k]])
        ans.append(dict(winding=winding[k].tolist(),trace=cj(np.trace(hol)),trace_squared=cj(np.trace(hol)**2),
                        approximate_hyperbolic_length=float(dist[i[k]]+weight[k]+dist[j[k]]),
                        determinant_error=float(abs(np.linalg.det(hol)-1))))
    return ans
