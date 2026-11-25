"""Generic-source reconstruction and section-relative Serre pairing.

No target fibre constraint is imposed. The fitted full Hopf differential is
recorded before its small complementary component is removed to identify the
candidate spectral elliptic quotient. This approximation is audited separately.
"""
import numpy as np,time
from connection_check import I,P,adj,inv,point_frame,hermitian_log_half
from seed_higgs import cj

def extract(data,p,tz,rings=2,degree=2):
    edge=data['edges'];tri=data['tri'];nn=len(p)
    frames=point_frame(p);g=inv(frames[edge[:,0]])@data['links']@frames[edge[:,1]]
    lookup={tuple(e):k for k,e in enumerate(edge)}
    dz=np.zeros(len(edge),complex)
    for a,b in [(0,1),(1,2),(2,0)]:
        ids=np.array([lookup[tuple(sorted((int(i),int(j))))] for i,j in tri[:,[a,b]]])
        sign=np.where(tri[:,a]<tri[:,b],1,-1)
        dz[ids]=sign*(tz[:,b]-tz[:,a])
    graph=[[] for _ in range(nn)]
    for k,(i,j) in enumerate(edge):
        graph[i].append((j,dz[k],g[k]));graph[j].append((i,-dz[k],inv(g[k])))
    phi=np.zeros((nn,2,2),complex);fitres=[]
    for i in range(nn):
        found={};frontier=[(i,0j,I)]
        for _ in range(rings):
            new=[]
            for j,w,T in frontier:
                for k,dz0,U in graph[j]:
                    if k==i or k in found:continue
                    found[k]=(w+dz0,T@U);new.append((k,w+dz0,T@U))
            frontier=new
        z=np.array([v[0] for v in found.values()]);T=np.array([v[1] for v in found.values()])
        v=hermitian_log_half(T);scale=max(max(abs(z)),1e-12);x,y=z.real/scale,z.imag/scale
        cols=[x,y,x*x,x*y,y*y]
        if degree>=3:cols.extend([x**3,x*x*y,x*y*y,y**3])
        X=np.stack(cols,axis=1);w=1/np.sqrt(.15+x*x+y*y)
        c=np.linalg.lstsq(w[:,None]*X,w[:,None]*v,rcond=1e-10)[0]
        phi[i]=np.einsum('a,aij->ij',.5*(c[0]-1j*c[1])/scale,P)
        fitres.append(np.linalg.norm(X@c-v)/max(np.linalg.norm(v),1e-14))
    vals,vecs=np.linalg.eig(phi);q=2*np.einsum('nij,nji->n',phi,phi)
    want=.5*np.sqrt(q);swap=abs(vals[:,0]-want)>abs(vals[:,1]-want)
    vals[swap]=vals[swap,::-1];vecs[swap]=vecs[swap,:,::-1]
    vecs/=np.linalg.norm(vecs,axis=1)[:,None,:]
    u,_,vh=np.linalg.svd(g);unit=u@vh;unit/=np.sqrt(np.linalg.det(unit))[:,None,None]
    return dict(phi=phi,q=q,values=vals,vectors=vecs,unitaries=unit,lookup=lookup,
                fit_median=float(np.median(fitres)),fit_95=float(np.quantile(fitres,.95)))

def measure(data,p,tz,F,metric,rings=2,degree=2):
    started=time.time();d=extract(data,p,tz,rings,degree);tri=data['tri'];nn=len(p)
    centers=tz.mean(axis=1);area=abs(((tz[:,1]-tz[:,0])*np.conj(tz[:,2]-tz[:,0])).imag)/2
    qraw=d['q'][tri].mean(axis=1)
    psi=F.differentials(centers)
    weight=area*np.exp(-2*metric.evaluate(centers,0))
    G=np.einsum('t,ti,tj->ij',weight,psi.conj(),psi);rhs=np.einsum('t,ti,t->i',weight,psi.conj(),qraw)
    c=np.linalg.solve(G,rhs);cf=c.copy();cf[2]=0
    # Orthogonal projection uses the full Gram matrix, not the algebraic
    # splitting between the first two and the third quadratic differential.
    a=np.linalg.solve(G[:2,:2],G[:2]@c);outside=c.copy();outside[:2]-=a
    fn=np.real(c.conj()@G@c);eps=np.sqrt(max(np.real(outside.conj()@G@outside),0)/max(fn,1e-300))
    eta=np.sqrt(np.sum(weight*abs(qraw-psi@c)**2)/max(np.sum(weight*abs(qraw)**2),1e-300))
    qfit=psi@cf;n=psi[:,2];lam=np.sqrt(qfit);beta=n/lam
    desired=[]
    for k in range(3):
        t=np.linspace(0,1,9)
        zz=centers[:,None]+t*(tz[:,k]-centers)[:,None]
        qq=(F.differentials(zz)@cf)
        phase=np.unwrap(np.angle(qq),axis=1)
        desired.append(np.sqrt(abs(qq[:,-1]))*np.exp(.5j*(phase[:,-1]-phase[:,0]))*lam/np.maximum(abs(lam),1e-300))
    desired=np.stack(desired,axis=1);actual=2*d['values'][tri,0]
    signs=(abs(actual-desired)>abs(actual+desired)).astype(int)
    qa=desired**2;wind=np.rint(np.angle(np.roll(qa,-1,axis=1)/qa).sum(axis=1)/(2*np.pi)).astype(int)
    ev=d['vectors'].copy();over=np.einsum('ni,ni->n',ev[:,:,1].conj(),ev[:,:,0]);ev[:,:,1]*=(over/np.maximum(abs(over),1e-300))[:,None]
    theta=[];overlaps=[]
    for a,b in [(0,1),(1,2),(2,0)]:
        ids=np.array([d['lookup'][tuple(sorted((int(i),int(j))))] for i,j in tri[:,[a,b]]])
        U=d['unitaries'][ids].copy();rev=tri[:,a]>tri[:,b];U[rev]=inv(U[rev])
        i,j=tri[:,a],tri[:,b];si,sj=signs[:,a],signs[:,b]
        up=np.einsum('ni,nij,nj->n',ev[i,:,si].conj(),U,ev[j,:,sj]);um=np.einsum('ni,nij,nj->n',ev[i,:,1-si].conj(),U,ev[j,:,1-sj])
        theta.append(np.angle(up/um));overlaps.append(np.minimum(abs(up),abs(um)))
    theta=np.stack(theta,axis=1);overlaps=np.stack(overlaps,axis=1)
    bedges=beta[:,None]*(np.roll(tz,-1,axis=1)-tz)
    integrand=np.sum(theta*np.roll(bedges,-1,axis=1)-np.roll(theta,-1,axis=1)*bedges,axis=1)/6
    det=((tz[:,1]-tz[:,0]).real*(tz[:,2]-tz[:,0]).imag-(tz[:,1]-tz[:,0]).imag*(tz[:,2]-tz[:,0]).real)
    integrand*=np.sign(det)
    good=np.isfinite(integrand)&(wind==0)&(np.min(overlaps,axis=1)>1e-4)
    value=np.sum(integrand[good]);norm=np.sqrt(np.sum(abs(beta[good])**2*area[good]))
    allbeta=psi/lam[:,None]
    allbedges=allbeta[:,None,:]*(np.roll(tz,-1,axis=1)-tz)[:,:,None]
    allintegrand=np.sum(theta[:,:,None]*np.roll(allbedges,-1,axis=1)-np.roll(theta,-1,axis=1)[:,:,None]*allbedges,axis=1)/6
    allintegrand*=np.sign(det)[:,None]
    allvalues=allintegrand[good].sum(axis=0);allnorms=np.sqrt(np.sum(abs(allbeta[good])**2*area[good,None],axis=0))
    return dict(extraction_degree=degree,extraction_rings=rings,hopf_coefficients=cj(c),pencil_defect=float(eps),holomorphic_fit_residual=float(eta),
                hopf_gram=cj(G),
                all_prym_pairings=cj(allvalues),all_prym_form_norms=allnorms.tolist(),
                fitted_hopf_norm=float(np.sqrt(fn)),discarded_algebraic_component=cj(c[2]),
                normal_serre_pairing=cj(value),normalized_fibre_defect=float(abs(value)/norm),
                beta_L2=float(norm),omitted_faces=int(sum(~good)),branch_faces=int(sum(wind!=0)),
                faces=len(tri),omitted_flat_area=float(sum(area[~good])),
                min_opposite_overlap=float(min(abs(over))),max_link_phase=float(abs(theta[good]).max()),
                eigenline_fit_median=d['fit_median'],eigenline_fit_95=d['fit_95'],seconds=time.time()-started)
