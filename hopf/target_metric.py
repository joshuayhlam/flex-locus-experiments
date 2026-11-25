"""Generic punctured-torus Liouville metric and analytic P2 interpolation.

No flex condition is used here.  F needs only F.T and F.p, where T has the
Torus interface in flex_wp.py.  The unknown is the P2 correction to the exact
periodic Green background.  Derivatives of the interpolant are analytic on
each triangle, including the background's exact first and second derivatives.
The resulting metric is continuous; its gradient need not match across edges.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse.linalg import spsolve
import flex_wp as fw


def background(F, z, derivatives=2):
    """-sum Green functions, with Cartesian gradient/Hessian."""
    z = np.asarray(z, complex)
    dz = F.T.center(z[..., None] - F.p)
    th = np.zeros(dz.shape, complex)
    dt = np.zeros_like(th) if derivatives else None
    ddt = np.zeros_like(th) if derivatives > 1 else None
    # The bound includes the maximum growth at |Im z| <= Im(tau)/2.
    for n in range(13):
        a = (2*n+1)*np.pi
        if n > 1 and np.exp(-np.pi*F.T.A*(n*n-.25))*a*a < 2e-18:
            break
        b = 2*(-1)**n*np.exp(1j*np.pi*F.T.tau*(n+.5)**2)
        sn = np.sin(a*dz)
        th += b*sn
        if derivatives:
            dt += b*a*np.cos(a*dz)
        if derivatives > 1:
            ddt -= b*a*a*sn
    val = -np.sum(np.log(abs(th))-np.pi*dz.imag**2/F.T.A, axis=-1)
    if not derivatives:
        return val
    d = dt/th
    grad = np.stack((-d.real.sum(axis=-1),
                     (d.imag+2*np.pi*dz.imag/F.T.A).sum(axis=-1)), axis=-1)
    if derivatives == 1:
        return val, grad
    h = ddt/th-d*d
    H = np.empty(z.shape+(2,2))
    H[...,0,0] = -h.real.sum(axis=-1)
    H[...,0,1] = H[...,1,0] = h.imag.sum(axis=-1)
    H[...,1,1] = h.real.sum(axis=-1)+len(F.p)*2*np.pi/F.T.A
    return val, grad, H


class TargetMetric:
    def __init__(self, F, M, u, stats):
        self.F, self.M, self.u, self.stats = F, M, np.asarray(u), stats
        self.hole_evaluations = 0
        self._prepare_locator()
        self._prepare_polynomials()
        total = self.u + M['u0nodes']
        self.cusp_constants = np.array([
            np.mean(np.exp(-total[ii])/M['rmin'])+np.log(M['rmin'])
            for ii in M['ring_ids']])
        self.stats['cusp_constants'] = self.cusp_constants.tolist()

    def _prepare_locator(self):
        M, T = self.M, self.F.T
        nv1 = int(np.max(M['tri'][:,:3]))+1
        z = M['z'][:nv1]
        tiles = [i+j*T.tau for i in (-1,0,1) for j in (-1,0,1)]
        zz = np.concatenate([z+off for off in tiles])
        self.locator = Delaunay(np.c_[zz.real,zz.imag])
        simplices = self.locator.simplices
        ids = simplices % nv1
        lookup = {tuple(sorted(row)):i for i,row in enumerate(M['tri'][:,:3])}
        self.locator_index = np.array([lookup.get(tuple(sorted(row)),-1) for row in ids])
        self.locator_origin = np.zeros(len(simplices),complex)
        good = np.flatnonzero(self.locator_index >= 0)
        wanted = M['tri'][self.locator_index[good],0]
        col = np.argmax(ids[good] == wanted[:,None], axis=1)
        self.locator_origin[good] = zz[simplices[good,col]]
        # Used only for roundoff points on cusp polygon boundaries.
        centers = M['tz'].mean(axis=1)
        center_copies = np.concatenate([centers+off for off in tiles])
        self.near_tree = cKDTree(np.c_[center_copies.real,center_copies.imag])
        self.near_ids = np.tile(np.arange(len(centers)),9)
        self.near_origins = np.concatenate([M['tz'][:,0]+off for off in tiles])

    def _prepare_polynomials(self):
        M = self.M
        g = M['grad']
        h = [4*g[:,i,:,None]*g[:,i,None,:] for i in range(3)]
        for i,j in [(0,1),(1,2),(2,0)]:
            h.append(4*(g[:,i,:,None]*g[:,j,None,:]+g[:,j,:,None]*g[:,i,None,:]))
        v = self.u[M['tri']]
        self.poly_hessian = np.einsum('ti,tixy->txy',v,np.stack(h,axis=1))
        self.poly_gradient = (3*v[:,0,None]*g[:,0]-v[:,1,None]*g[:,1]
            -v[:,2,None]*g[:,2]+4*v[:,3,None]*g[:,1]+4*v[:,5,None]*g[:,2])
        self.poly_constant = v[:,0]

    def locate(self, w):
        """Return flattened triangle IDs, lifted first vertices, periodic w."""
        z = self.F.T.z(self.F.T.uv(np.asarray(w,complex).ravel()) % 1)
        simplex = self.locator.find_simplex(np.c_[z.real,z.imag],tol=1e-11)
        idx = self.locator_index[np.maximum(simplex,0)].copy()
        idx[simplex < 0] = -1
        origin = self.locator_origin[np.maximum(simplex,0)].copy()
        bad = np.flatnonzero(idx < 0)
        if len(bad):
            _, near = self.near_tree.query(np.c_[z[bad].real,z[bad].imag],k=16)
            ni = self.near_ids[near]
            dd = z[bad,None]-self.near_origins[near]
            delta = np.stack((dd.real,dd.imag),axis=-1)
            bary = np.einsum('bkix,bkx->bki',self.M['grad'][ni],delta)
            bary[:,:,0] += 1
            quality = bary.min(axis=-1)
            col = np.argmax(quality,axis=1)
            accepted = quality[np.arange(len(bad)),col] > -2e-9
            bb = bad[accepted]; cc = col[accepted]
            idx[bb] = ni[accepted,cc]
            origin[bb] = self.near_origins[near[accepted,cc]]
        return idx,origin,z

    def evaluate(self,w,derivatives=2):
        """Return U, or (U,grad U), or (U,grad U,Hess U).

        Points strictly inside a truncated cusp use its fitted radial metric.
        This is an asymptotic continuation, not an additional PDE solution.
        """
        shape = np.shape(w)
        idx,origin,z = self.locate(w)
        good = idx >= 0
        safe = np.maximum(idx,0)
        dz = z-origin
        d = np.c_[dz.real,dz.imag]
        H = self.poly_hessian[safe]
        gd = np.einsum('nxy,ny->nx',H,d)
        val = self.poly_constant[safe]+np.sum(self.poly_gradient[safe]*d,axis=1)+.5*np.sum(d*gd,axis=1)
        bg = background(self.F,z,derivatives)
        val += bg if derivatives == 0 else bg[0]
        if derivatives:
            grad = self.poly_gradient[safe]+gd+bg[1]
        if derivatives > 1:
            H = H+bg[2]
        if not np.all(good):
            bad = np.flatnonzero(~good)
            self.hole_evaluations += len(bad)
            dc = self.F.T.center(z[bad,None]-self.F.p)
            nearest = np.argmin(abs(dc),axis=1)
            dc = dc[np.arange(len(bad)),nearest]
            r = np.maximum(abs(dc),1e-15)
            L = np.maximum(self.cusp_constants[nearest]-np.log(r),1e-12)
            val[bad] = -np.log(r)-np.log(L)
            if derivatives:
                unit = np.c_[dc.real,dc.imag]/r[:,None]
                fp = (-1+1/L)/r
                grad[bad] = fp[:,None]*unit
            if derivatives > 1:
                fpp = (1-1/L+1/L**2)/r**2
                H[bad] = ((fpp-fp/r)[:,None,None]*unit[:,:,None]*unit[:,None,:]
                    +(fp/r)[:,None,None]*np.eye(2))
        val = val.reshape(shape)
        if not derivatives:
            return val
        grad = grad.reshape(shape+(2,))
        if derivatives == 1:
            return val,grad
        return val,grad,H.reshape(shape+(2,2))

    __call__ = evaluate

    def checks(self):
        """Check nodal/midpoint reproduction and polynomial derivatives."""
        M = self.M
        total = self.u+M['u0nodes']
        vals = self.evaluate(M['z'],0)
        nodal = float(np.max(abs(vals-total)))
        z = M['tz'] @ np.array([.23,.31,.46])
        z = z[::max(1,len(z)//100)]
        U,g,H = self.evaluate(z,2)
        # Choose step much smaller than distance to triangle edges.
        eps = min(1e-7,M['rmin']*1e-4)
        numeric_g = np.stack([(self.evaluate(z+eps*d,0)-self.evaluate(z-eps*d,0))/(2*eps)
                              for d in (1,1j)],axis=-1)
        numeric_H = np.stack([(self.evaluate(z+eps*d,1)[1]-self.evaluate(z-eps*d,1)[1])/(2*eps)
                              for d in (1,1j)],axis=-1)
        return {'nodal_midpoint_max_error':nodal,
            'gradient_relative_error':float(np.max(abs(numeric_g-g)/(1+abs(g)))),
            'hessian_relative_error':float(np.max(abs(numeric_H-H)/(1+abs(H)))),
            'interpolation_hole_evaluations':self.hole_evaluations}


def solve_metric(F,N=24,rmin=1e-3,ntheta=24,ratio=1.35,quad=4,cache=None,
                 tolerance=2e-10,maxiter=35):
    """Solve an unconstrained generic hyperbolic target metric.

    Cache stores the correction and JSON metadata.  Meshes are deterministically
    rebuilt and verified against cached vertex/triangle arrays before reuse.
    """
    start = time.time()
    fw.quadrature_order(quad)
    M = fw.mesh(F,N,rmin,ntheta,ratio)
    fw.quadratic_elements(F,M)
    fw.cusp_background(F,M,'green')
    M['bary'] = fw.BARY.copy()
    M['weights'] = fw.BW.copy()
    settings = dict(N=N,rmin=rmin,ntheta=ntheta,ratio=ratio,quad=quad,
                    tau=[F.T.tau.real,F.T.tau.imag],
                    punctures=np.c_[F.p.real,F.p.imag].tolist())
    saved = Path(cache).with_suffix('.npz') if cache else None
    if saved is not None and saved.exists():
        with np.load(saved) as d:
            old = json.loads(str(d['stats']))
            if old['settings'] == settings and np.array_equal(d['tri'],M['tri']) and np.max(abs(d['z']-M['z'])) < 1e-13:
                metric = TargetMetric(F,M,d['u'].copy(),old)
                metric.stats['loaded_from_cache'] = True
                return metric
    distance = np.min(np.stack([abs(F.T.center(M['z']-p)) for p in F.p]),axis=0)
    u = -np.log(distance)-np.log(np.maximum(.8-np.log(distance),.1))+.4-M['u0nodes']
    history = []
    for it in range(maxiter):
        E,r,H,area,tail = fw.nonlinear(M,u)
        history.append([float(E),float(np.max(abs(r)))])
        if np.max(abs(r)) < tolerance:
            break
        du = spsolve(H,-r)
        step = 1.
        while step > 1e-8:
            ne,nr,_,_ = fw.nonlinear(M,u+step*du,False)
            if ne <= E+1e-4*step*(r@du)+1e-11:
                break
            step *= .5
        if step <= 1e-8:
            raise RuntimeError('Liouville Newton line search failed')
        u += step*du
    E,r,area,tail = fw.nonlinear(M,u,False)
    residual = float(np.max(abs(r)))
    if residual > max(tolerance*5,1e-8):
        raise RuntimeError(f'Liouville solve did not converge: {residual}')
    stats = dict(settings=settings,vertices=M['nv'],triangles=len(M['tri']),
        newton_residual=residual,iterations=it,history=history,
        hyperbolic_area_truncated=area,cusp_tail_area_robin=tail,
        gauss_bonnet_error=area+tail-len(F.p)*2*np.pi,
        seconds=time.time()-start,loaded_from_cache=False)
    metric = TargetMetric(F,M,u,stats)
    stats['interpolation_checks'] = metric.checks()
    if saved is not None:
        saved.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(saved,u=u,z=M['z'],tri=M['tri'],stats=json.dumps(stats))
        saved.with_suffix('.json').write_text(json.dumps(stats,indent=2))
    return metric


if __name__ == '__main__':
    from second_fundamental import Ambient, FlexChart
    F = Ambient(FlexChart(.23+1.08j,5+2j).x)
    metric = solve_metric(F,N=12,rmin=.003,ntheta=16,quad=4)
    print(json.dumps(metric.stats,indent=2))
