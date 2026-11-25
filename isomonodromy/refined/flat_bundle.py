"""Cellular reconstruction of an exactly flat bundle with parabolic cusps.

Primal and dual spanning trees leave four free generators. All face relations
are then imposed as free-group identities. Only the three peripheral trace
constraints are projected; no flex or spectral-line condition is imposed.
"""
from collections import deque
import json,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from connection_check import inv,exp2,I,P
from seed_higgs import cj

def reduce_word(*words):
    out=[]
    for w in words:
        for a in w:
            if out and out[-1]==-a:out.pop()
            else:out.append(int(a))
    return tuple(out)

def wi(w):return tuple(-a for a in w[::-1])

def value(w,G):
    v=I.copy()
    for a in w:v=v@(G[a-1] if a>0 else inv(G[-a-1]))
    return v

def reconstruct(data,F):
    start=time.time();edges=data['edges'];tri=data['tri'];z=data['z'];i,j=edges.T;nn=len(z);nf=len(tri)
    lookup={tuple(e):k for k,e in enumerate(edges)}
    length=abs(F.T.center(z[j]-z[i]))
    if 'node_U' in data:length*=np.exp(.5*(data['node_U'][i]+data['node_U'][j]))
    graph=coo_matrix((np.r_[length,length],(np.r_[i,j],np.r_[j,i])),shape=(nn,nn)).tocsr()
    sep=np.min([abs(F.T.center(z-p)) for p in F.p],axis=0);base=int(np.argmax(sep))
    dist,pred=dijkstra(graph,indices=base,return_predecessors=True)
    tree=set();frames=np.broadcast_to(I,(nn,2,2)).copy()
    for v in np.argsort(dist):
        if v==base:continue
        u=int(pred[v]);k=lookup[tuple(sorted((u,int(v))))];tree.add(k)
        g=data['links'][k] if u<v else inv(data['links'][k]);frames[v]=frames[u]@g
    face_edges=[];adjacency=[[] for _ in range(len(edges))]
    boundary_orient={}
    for t,row in enumerate(tri):
        ee=[]
        for a,b in [(0,1),(1,2),(2,0)]:
            u,v=int(row[a]),int(row[b]);k=lookup[tuple(sorted((u,v)))];s=1 if u<v else -1
            ee.append((k,s));adjacency[k].append(t)
            if k in boundary_orient:del boundary_orient[k]
            else:boundary_orient[k]=(u,v)
        face_edges.append(ee)
    dual=[[] for _ in range(nf+1)];root=nf
    for k,tt in enumerate(adjacency):
        if k in tree:continue
        a,b=tt if len(tt)==2 else (tt[0],root)
        dual[a].append((b,k));dual[b].append((a,k))
    parent_edge=np.full(nf,-1,int);seen={root};queue=deque([root]);order=[];dual_tree=set()
    while queue:
        a=queue.popleft()
        for b,k in dual[a]:
            if b in seen:continue
            seen.add(b);parent_edge[b]=k;dual_tree.add(k);order.append(b);queue.append(b)
    if len(order)!=nf:raise RuntimeError('Dual complement is disconnected')
    free=sorted(set(range(len(edges)))-tree-dual_tree)
    if len(free)!=4:raise RuntimeError(f'Expected four generators, got {len(free)}')
    words=[None]*len(edges)
    for k in tree:words[k]=()
    for a,k in enumerate(free):words[k]=(a+1,)
    for t in order[::-1]:
        ee=face_edges[t];k=parent_edge[t];m=next(a for a,(kk,ss) in enumerate(ee) if kk==k)
        other=[]
        for a in [(m+1)%3,(m+2)%3]:
            kk,ss=ee[a];w=words[kk]
            if w is None:raise RuntimeError('Unresolved face relation')
            other.append(w if ss==1 else wi(w))
        w=wi(reduce_word(*other));words[k]=w if ee[m][1]==1 else wi(w)
    # Every relator must vanish as a word, not just as a floating-point matrix.
    for ee in face_edges:
        if reduce_word(*[words[k] if s==1 else wi(words[k]) for k,s in ee]):raise RuntimeError('Nontrivial face word')
    nxt={u:(v,k) for k,(u,v) in boundary_orient.items()};remaining=set(nxt);peripheral=[]
    while remaining:
        first=next(iter(remaining));u=first;w=()
        while True:
            remaining.remove(u);v,k=nxt[u];w=reduce_word(w,words[k] if u<v else wi(words[k]));u=v
            if u==first:break
        peripheral.append(w)
    if len(peripheral)!=3:raise RuntimeError('Expected three boundary components')
    G=np.array([frames[edges[k,0]]@data['links'][k]@inv(frames[edges[k,1]]) for k in free])
    G/=np.sqrt(np.linalg.det(G))[:,None,None];G0=G.copy()
    def residual(G):return np.array([np.trace(value(w,G))+2 for w in peripheral])
    history=[]
    for it in range(12):
        r=residual(G);history.append(float(abs(r).max()))
        if abs(r).max()<2e-12:break
        J=np.empty((3,12),complex);eps=1e-6
        for k in range(4):
            for a in range(3):
                plus=G.copy();minus=G.copy();plus[k]=G[k]@exp2(eps*P[a]);minus[k]=G[k]@exp2(-eps*P[a])
                J[:,3*k+a]=(residual(plus)-residual(minus))/(2*eps)
        delta=np.linalg.lstsq(J,-r,rcond=1e-10)[0].reshape(4,3)
        step=1.
        while step>1e-5:
            Gn=G@exp2(step*np.einsum('na,aij->nij',delta,P))
            if np.linalg.norm(residual(Gn))<np.linalg.norm(r):break
            step*=.5
        G=Gn
    if abs(residual(G)).max()>1e-9:raise RuntimeError('Peripheral projection failed')
    cache={w:value(w,G) for w in set(words)}
    rootlinks=np.array([cache[w] for w in words])
    links=inv(frames[i])@rootlinks@frames[j]
    face_error=[]
    for ee in face_edges:
        prod=I.copy()
        for k,s in ee:prod=prod@(links[k] if s==1 else inv(links[k]))
        face_error.append(np.linalg.norm(prod-I))
    change=links@inv(data['links'])
    changes=np.linalg.norm(change-I,axis=(1,2))
    report=dict(generators=cj(G),generator_edges=free,primal_root=base,peripheral_words=peripheral,
        peripheral_traces=cj(np.array([np.trace(value(w,G)) for w in peripheral])),
        generator_trace_squared=cj(np.trace(G,axis1=-2,axis2=-1)**2),
        initial_peripheral_trace_error=history[0],final_peripheral_trace_error=float(abs(residual(G)).max()),
        peripheral_projection_history=history,generator_change=float(np.linalg.norm(G-G0)),
        edge_change_max=float(max(changes)),edge_change_rms=float(np.sqrt(np.mean(changes**2))),
        flat_face_error_max=float(max(face_error)),flat_face_error_median=float(np.median(face_error)),
        maximum_word_length=max(map(len,words)),distinct_words=len(cache),seconds=time.time()-start)
    return links,report,dict(words=words,frames=frames,generators=G,peripheral_words=peripheral)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    import flex_wp as fw
    from seed_higgs import BASES
    ap=argparse.ArgumentParser();ap.add_argument('run');a=ap.parse_args();p=Path(a.run)
    seed=json.loads((p/'seed.json').read_text());saved=np.load(p/'fixed_connection.npz')
    data={k:saved[k] for k in ['edges','links','tri','tz','z']};F=fw.Flex(*BASES[seed['base']])
    U=saved['u'].copy()
    for q in F.p:U-=F.T.green(data['z']-q)[0]
    data['node_U']=U
    links,report,extra=reconstruct(data,F)
    (p/'flat_projection.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(p/'flat_connection.npz',links=links,edges=data['edges'],tri=data['tri'],tz=data['tz'],z=data['z'],frames=extra['frames'],generators=extra['generators'])
    print(json.dumps(report,indent=2))
