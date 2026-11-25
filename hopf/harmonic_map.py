"""Unconstrained P2 harmonic maps between marked, triply punctured tori.

Minimize antiholomorphic energy, equal to Dirichlet energy up to degree/area.
The flex pencil enters only the final measurement, never the map solve.
"""
import time
import numpy as np
import flex_wp as fw
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class HarmonicMap:
    def __init__(self, source_metric, target_metric, warp):
        self.source, self.target = source_metric, target_metric
        M = self.M = source_metric.M
        self.shape = M['shape']
        self.grad = M['gradq']
        self.weight = M['area'][:,None]*M['weights'][None,:]
        self.zq = M['tz'] @ M['bary'].T
        self.nv = M['nv']
        tx,ty = source_metric.F.T.tau,target_metric.F.T.tau
        self.ax,self.ay = 1+0j,(ty-tx.real)/tx.imag
        self.affine = lambda z,ay=self.ay: z.real+ay*z.imag
        self.d0 = warp(M['z'])-self.affine(M['z'])
        self.affq = self.affine(self.zq)
        # Every boundary node is constrained to a conformal affine cusp germ.
        bound = np.unique(M['be'])
        interior = np.setdiff1d(np.arange(self.nv),bound)
        nint = len(interior)
        rows = np.ravel(np.c_[2*interior,2*interior+1]).tolist()
        cols = np.arange(2*nint).tolist()
        vals = np.ones(2*nint).tolist()
        dz = source_metric.F.T.center(M['z'][bound,None]-source_metric.F.p)
        near = np.argmin(abs(dz),axis=1)
        dz = dz[np.arange(len(bound)),near]
        for node,j,d in zip(bound,near,dz):
            rows.extend([2*node,2*node,2*node+1,2*node+1])
            cols.extend([2*nint+2*j,2*nint+2*j+1,2*nint+2*j,2*nint+2*j+1])
            vals.extend([d.real,-d.imag,d.imag,d.real])
        self.P = coo_matrix((vals,(rows,cols)),shape=(2*self.nv,2*nint+6)).tocsr()
        self.nint=nint
        self.v=np.zeros(self.P.shape[1])
        dof=np.stack((2*M['tri'],2*M['tri']+1),axis=-1).reshape(-1,12)
        self.dof=dof
        self.rows=np.broadcast_to(dof[:,:,None],(len(dof),12,12)).ravel()
        self.cols=np.broadcast_to(dof[:,None,:],(len(dof),12,12)).ravel()
        gx,gy=self.grad[...,0],self.grad[...,1]
        self.B=np.empty(gx.shape+(2,2))
        self.B[...,0,0]=gx/2; self.B[...,0,1]=gy/2
        self.B[...,1,0]=-gy/2; self.B[...,1,1]=gx/2

    def fields(self,v):
        dd=(self.P@v).reshape(-1,2)
        d=self.d0+dd[:,0]+1j*dd[:,1]
        local=d[self.M['tri']]
        w=self.affq+np.einsum('qi,ti->tq',self.shape,local)
        dw=np.einsum('tqix,ti->tqx',self.grad,local)
        wx=self.ax+dw[...,0]; wy=self.ay+dw[...,1]
        wz=(wx-1j*wy)/2; wb=(wx+1j*wy)/2
        return d,w,wz,wb

    def objective(self,v,derivatives=False):
        d,w,wz,wb=self.fields(v)
        if derivatives:
            U,g=self.target.evaluate(w,1)
        else:
            U=self.target.evaluate(w,0)
        if np.max(U)>100:
            return np.inf
        rho=np.exp(2*U)
        b=np.stack((wb.real,wb.imag),axis=-1)
        E=float(np.sum(self.weight*rho*abs(wb)**2))
        if not derivatives:
            return E
        J=self.B+self.shape[None,:,:,None,None]*g[:,:,None,:,None]*b[:,:,None,None,:]
        J=J.reshape(J.shape[:2]+(12,2))
        wr=self.weight*rho
        localg=2*np.einsum('tq,tqir,tqr->ti',wr,J,b,optimize=True)
        fullg=np.bincount(self.dof.ravel(),weights=localg.ravel(),minlength=2*self.nv)
        localH=2*np.einsum('tq,tqir,tqjr->tij',wr,J,J,optimize=True)
        H=coo_matrix((localH.ravel(),(self.rows,self.cols)),shape=(2*self.nv,2*self.nv)).tocsr()
        return E,self.P.T@fullg,(self.P.T@H@self.P).tocsc()

    def gradient_check(self,seed=102):
        rng=np.random.default_rng(seed)
        v=self.v.copy()
        direction=rng.standard_normal(len(v));direction/=np.linalg.norm(direction)
        E,g,H=self.objective(v,True)
        ans=[]
        for eps in (1e-5,1e-6,1e-7):
            numeric=(self.objective(v+eps*direction)-self.objective(v-eps*direction))/(2*eps)
            exact=float(g@direction)
            ans.append(dict(eps=eps,numeric=numeric,analytic=exact,relative=abs(numeric-exact)/max(abs(exact),1e-15)))
        return ans

    def solve(self,maxiter=50,tolerance=2e-9,verbose=False):
        start=time.time(); history=[]; v=self.v.copy()
        initial_fallback=self.target.hole_evaluations
        for it in range(maxiter):
            E,g,H=self.objective(v,True)
            dv=spsolve(H,-g)
            decrement=float(np.sqrt(max(-g@dv,0)))
            history.append(dict(iteration=it,energy=E,gradient_inf=float(np.max(abs(g))),decrement=decrement,max_step=float(np.max(abs(dv)))))
            if verbose: print('map',it,E,decrement,flush=True)
            if decrement<tolerance*max(np.sqrt(E),1e-4):
                break
            step=1.
            for backtrack in range(24):
                Et=self.objective(v+step*dv)
                # At tiny decrements the predicted decrease is below summation
                # roundoff. Permit only a relative 1e-14 energy slack there;
                # the next iteration still checks the actual gradient.
                slack=1e-14*max(E,1e-8) if decrement<1e-7*max(np.sqrt(E),1e-4) else 0.
                if np.isfinite(Et) and Et<=E+1e-4*step*(g@dv)+slack: break
                step*=.5
            history[-1]['step']=step
            if backtrack==23:
                history[-1]['line_search_failed']=True
                break
            v+=step*dv
        self.v=v
        d,w,wz,wb=self.fields(v)
        self.d=d
        self.stats=dict(iterations=len(history),history=history,seconds=time.time()-start,
            converged=bool(history[-1]['decrement']<tolerance*max(np.sqrt(history[-1]['energy']),1e-4)),
            cusp_slopes=np.c_[1+v[-6::2],v[-5::2]].tolist(),
            min_jacobian=float(np.min(abs(wz)**2-abs(wb)**2)),
            max_beltrami=float(np.max(abs(wb)/abs(wz))),
            hole_evaluations=self.target.hole_evaluations-initial_fallback)
        return self.stats

    def measure(self,W_coeff):
        d,w,wz,wb=self.fields(self.v)
        Uy=self.target.evaluate(w,0)
        q=np.exp(2*Uy)*wz*np.conj(wb)
        Ux=self.source.evaluate(self.zq,0)
        F=self.source.F.differentials(self.zq)
        # Condition the basis while retaining its exact W subspace.
        weight=self.weight*np.exp(-2*Ux)
        norms=np.sqrt(np.einsum('tq,tqi,tqi->i',weight,F.conj(),F).real)
        F=F/norms
        G=np.einsum('tq,tqi,tqj->ij',weight,F.conj(),F)
        rhs=np.einsum('tq,tqi,tq->i',weight,F.conj(),q)
        c=np.linalg.solve(G,rhs)
        fit=np.einsum('tqi,i->tq',F,c)
        V=norms[:,None]*W_coeff
        a=np.linalg.solve(V.conj().T@G@V,V.conj().T@G@c)
        outside=c-V@a
        qn=float(np.sum(weight*abs(q)**2))
        fn=float(np.real(c.conj()@G@c))
        on=float(np.real(outside.conj()@G@outside))
        fiterror=float(np.sqrt(np.sum(weight*abs(q-fit)**2)/max(qn,1e-300)))
        # Restore the leading asymptotic WP tail of the FULL fitted differential.
        # fw stores the conjugate Gram convention; transpose to our convention.
        tail,_=fw.cusp_gram_tail(self.source.F,self.M,self.source.u,norms)
        Gfull=G+tail.T
        afull=np.linalg.solve(V.conj().T@Gfull@V,V.conj().T@Gfull@c)
        ofull=c-V@afull
        fullnorm=float(np.real(c.conj()@Gfull@c))
        fulloutside=float(np.real(ofull.conj()@Gfull@ofull))
        return dict(hopf_norm=float(np.sqrt(qn)),fitted_hopf_norm=float(np.sqrt(max(fn,0))),
            pencil_defect=float(np.sqrt(max(on,0)/max(fn,1e-300))),
            absolute_pencil_defect=float(np.sqrt(max(on,0))),
            holomorphic_fit_residual=fiterror,gram_condition=float(np.linalg.cond(G)),
            tail_corrected_pencil_defect=float(np.sqrt(max(fulloutside,0)/max(fullnorm,1e-300))),
            tail_corrected_fitted_norm=float(np.sqrt(max(fullnorm,0))),
            fitted_norm_squared_tail_fraction=float(np.real(c.conj()@tail.T@c)/max(fullnorm,1e-300)),
            coefficients=np.c_[(c/norms).real,(c/norms).imag].tolist(),
            outside_coefficients=np.c_[(outside/norms).real,(outside/norms).imag].tolist())
