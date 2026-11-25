"""Marked source/target tori for an unconstrained nearby-Fuchsian experiment.

The target coordinates are computed algebraically before any harmonic solve.
The control targets perturb one marked point out of the flex surface.  No
flex equation, normal direction, or W-pencil enters the harmonic equation.
"""
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import numpy as np

from flex_wp import Flex
from second_fundamental import Ambient, FlexChart, cj


def separations(ambient):
    p = np.r_[0j, ambient.x[1:]]
    return np.array([min(abs(ambient.T.center(a-b)) for j, b in enumerate(p)
                         if i != j) for i, a in enumerate(p)])


def marked_jacobian(chart, u, scale=1.0):
    """Real central differences of the holomorphic marked parameter map."""
    out = []
    for j, h in enumerate(np.array([2e-5, 2e-4])*scale):
        d = np.zeros(2, complex)
        d[j] = h
        out.append((chart.at(u+d)-chart.at(u-d))/(2*h))
    return np.stack(out, axis=1)


def pencil_coefficients(chart):
    """Columns express the original pencil in normalized Ambient's Q basis.

    Flex.p[0] is the source origin before normalizing the first mark to zero.
    This translation is essential: the raw and normalized curves have different
    coordinate origins even though the differential dz is unchanged.
    """
    A = Ambient(chart.x)
    uv = np.array([[.13,.23], [.31,.41], [.63,.29], [.77,.71],
                   [.23,.67], [.48,.81], [.84,.46], [.54,.16]])
    z = A.T.z(uv)
    Q = A.differentials(z)
    W = chart.F.differentials(z+chart.F.p[0])[..., :2]
    C = np.linalg.lstsq(Q, W, rcond=None)[0]
    err = float(np.linalg.norm(Q@C-W)/np.linalg.norm(W))
    return C, err


class AlgebraicFlexResidual:
    """A local, independently evaluated algebraic equation for the flex locus.

    Choose the source branch of the translated Weierstrass origin and of the
    three half-preimages R_i under [-2]. Their sum is zero, hence they lie on
    a line Y=mX+b. The flex condition on this line is

       12 m b^2 + 24 b g2 - g2 m^3 - 36 g3 m = 0.

    Indeed s=m^2/12 and t=-b-g2/m turn this into the Hessian equation
    t^2=g2*s+3*g3+g2^2/(12*s). Evaluation below takes only the ambient marked
    curve, never its proposed flex parameters or its original construction.
    """
    def __init__(self, chart):
        T=chart.F.T
        w=np.r_[0j,chart.x[1:]]
        p0=chart.F.p[0]
        self.origin_period=np.rint(T.uv(3*p0+w.sum())).astype(int)
        xs=np.array([complex(*a) for a in chart.F.metadata['critical_x']])
        ys=np.array([complex(*a) for a in chart.F.metadata['critical_y']])
        R=np.array([T.invwp(a,b) for a,b in zip(xs,ys)])
        self.half_periods=np.rint(T.uv(2*R+p0+w)).astype(int)

    def __call__(self, x):
        A=Ambient(x)
        T=A.T
        w=np.r_[0j,A.x[1:]]
        origin=(T.z(self.origin_period)-w.sum())/3
        R=(-(origin+w)+T.z(self.half_periods))/2
        X,Y=T.wp(R)
        pts=np.stack((X,Y,np.ones(3)),axis=-1)
        line=np.cross(pts[0],pts[1])
        m=-line[0]/line[1]
        b=-line[2]/line[1]
        terms=np.array([12*m*b*b,24*b*T.g2,-T.g2*m**3,-36*T.g3*m])
        residual=terms.sum()
        return dict(cayleyan_polynomial=cj(residual),
                    cayleyan_relative_residual=float(abs(residual)/np.sum(abs(terms))),
                    half_preimage_collinearity_error=float(np.max(abs(pts@line))/(np.linalg.norm(line)*np.max(np.linalg.norm(pts,axis=1)))),
                    recovered_s=cj(m*m/12),recovered_t=cj(-b-T.g2/m),
                    origin_translation=cj(origin))


class ReferenceWarp:
    """Smooth marking with lattice equivariance and exact cusp translation.

    In a small disk around each source puncture this map is z -> z+constant.
    Away from larger, disjoint disks it is the real affine lattice map.
    Its derivatives use source Cartesian coordinates (Re z, Im z).
    """
    def __init__(self, source_x, target_x, inner_fraction=.18, outer_fraction=.42):
        self.source = Ambient(source_x)
        self.target = Ambient(target_x)
        self.T = self.source.T
        self.source_p = np.r_[0j, self.source.x[1:]]
        self.target_p = np.r_[0j, self.target.x[1:]]
        self.dtau = self.target.x[0]-self.source.x[0]
        self.B = np.array([[0., self.dtau.real/self.T.A],
                           [0., self.dtau.imag/self.T.A]])
        self.A = np.eye(2)+self.B
        # Also keep patches inside the embedded disk of the flat torus.
        sys = min(abs(i+j*self.T.tau) for i in range(-2,3) for j in range(-2,3)
                  if i or j)
        self.sep = np.minimum(separations(self.source), sys)
        self.inner = inner_fraction*self.sep
        self.outer = outer_fraction*self.sep
        self.displacements = self.target_p-self.affine(self.source_p)

    def affine(self, z):
        z = np.asarray(z, complex)
        return z+self.dtau*z.imag/self.T.A

    def _patch(self, z, i):
        delta = self.T.center(np.asarray(z, complex)-self.source_p[i])
        r = abs(delta)
        width = self.outer[i]-self.inner[i]
        t = np.clip((r-self.inner[i])/width, 0., 1.)
        chi = 1-10*t**3+15*t**4-6*t**5
        cr = (-30*t**2+60*t**3-30*t**4)/width
        crr = (-60*t+180*t**2-120*t**3)/width**2
        vec = np.stack((delta.real, delta.imag), axis=-1)
        e = vec/np.maximum(r, 1e-300)[..., None]
        grad = cr[..., None]*e
        ee = e[..., :, None]*e[..., None, :]
        hess = crr[..., None, None]*ee+(cr/np.maximum(r,1e-300))[...,None,None]*(np.eye(2)-ee)
        c = self.displacements[i]-self.dtau*delta.imag/self.T.A
        cv = np.stack((c.real, c.imag), axis=-1)
        return chi, grad, hess, c, cv

    def __call__(self, z):
        out = self.affine(z)
        for i in range(3):
            chi, _, _, c, _ = self._patch(z, i)
            out = out+chi*c
        return out

    def jacobian(self, z):
        z = np.asarray(z, complex)
        J = np.broadcast_to(self.A, z.shape+(2,2)).copy()
        for i in range(3):
            chi, grad, _, _, cv = self._patch(z, i)
            J += cv[..., :, None]*grad[..., None, :]-chi[...,None,None]*self.B
        return J

    def hessian(self, z):
        z = np.asarray(z, complex)
        H = np.zeros(z.shape+(2,2,2))
        for i in range(3):
            _, grad, hess, _, cv = self._patch(z, i)
            H += cv[..., :, None, None]*hess[..., None, :, :]
            H -= self.B[..., :, :, None]*grad[..., None, None, :]
            H -= self.B[..., :, None, :]*grad[..., None, :, None]
        return H

    def audit(self):
        grid = (np.arange(91)+.37)/91
        uv = np.stack(np.meshgrid(grid, grid, indexing='ij'),axis=-1)
        z = self.T.z(uv).ravel()
        theta = np.linspace(0,2*np.pi,181,endpoint=False)
        # Resolve the entire transition collar independently of the torus grid.
        for p, ri, ro in zip(self.source_p,self.inner,self.outer):
            rr = np.linspace(ri,ro,61)
            z = np.r_[z,(p+rr[:,None]*np.exp(1j*theta)).ravel()]
        det = np.linalg.det(self.jacobian(z))
        sample = self.T.z(np.array([[.23,.37],[.51,.69],[.81,.17]]))
        sample = np.r_[sample,self.source_p+(.5*(self.inner+self.outer))*np.exp(.73j)]
        eps = 1e-6
        Jfd = np.stack(((self(sample+eps)-self(sample-eps))/(2*eps),
                       (self(sample+1j*eps)-self(sample-1j*eps))/(2*eps)),axis=-1)
        Jfd = np.stack((Jfd.real,Jfd.imag),axis=-2)
        Hfd = np.stack(((self.jacobian(sample+eps)-self.jacobian(sample-eps))/(2*eps),
                        (self.jacobian(sample+1j*eps)-self.jacobian(sample-1j*eps))/(2*eps)),axis=-1)
        collar = np.concatenate([p+(.5*ri)*np.exp(1j*theta[:9])
                                 for p,ri in zip(self.source_p,self.inner)])
        period_error = max(np.max(abs(self(sample+1)-self(sample)-1)),
                           np.max(abs(self(sample+self.T.tau)-self(sample)-self.target.T.tau)))
        return dict(min_sampled_jacobian_determinant=float(det.min()),
                    max_sampled_jacobian_determinant=float(det.max()),
                    puncture_image_error=float(np.max(abs(self(self.source_p)-self.target_p))),
                    lattice_equivariance_error=float(period_error),
                    cusp_translation_jacobian_error=float(np.max(abs(self.jacobian(collar)-np.eye(2)))),
                    jacobian_finite_difference_error=float(np.max(abs(self.jacobian(sample)-Jfd))),
                    hessian_finite_difference_error=float(np.max(abs(self.hessian(sample)-Hfd))),
                    inner_radii=self.inner.tolist(),outer_radii=self.outer.tolist(),
                    marking_displacements=cj(self.displacements))


@dataclass
class TargetCase:
    name: str
    ambient: Ambient
    u: np.ndarray
    kind: str
    amplitude: float
    warp: ReferenceWarp
    diagnostics: dict = field(default_factory=dict)

    @property
    def x(self):
        return self.ambient.x

    def metadata(self):
        return dict(name=self.name,kind=self.kind,amplitude=self.amplitude,
                    marked_coordinates=cj(self.x), flex_parameters=cj(self.u),
                    diagnostics=self.diagnostics,warp=self.warp.audit())


@dataclass
class BaseCase:
    name: str
    chart: FlexChart
    source: Ambient
    W_coeff: np.ndarray
    targets: dict
    diagnostics: dict

    @property
    def x(self):
        return self.source.x

    def metadata(self):
        return dict(name=self.name,parameters=cj(self.chart.u),marked_coordinates=cj(self.x),
                    W_coeff=cj(self.W_coeff),diagnostics=self.diagnostics,
                    targets={k:v.metadata() for k,v in self.targets.items()})


def make_cases(include_identity=True):
    """Two generic sources, two flex amplitudes, and matched transverse controls."""
    configs = {
        'A':(.23+1.08j,5+2j,.002+.001j,.04-.03j,-.001+.002j,.02+.025j,.004-.003j),
        'B':(.07+.94j,-4+3j,-.0015+.0025j,.03+.04j,.002+.001j,-.025+.02j,-.004+.003j),
    }
    cases = {}
    for label,(tau,s,dt,ds,dt2,ds2,control_delta) in configs.items():
        chart=FlexChart(tau,s)
        source=Ambient(chart.x)
        C,fit=pencil_coefficients(chart)
        D=marked_jacobian(chart,chart.u)
        Dc=marked_jacobian(chart,chart.u,.5)
        residual=AlgebraicFlexResidual(chart)
        directions=np.array([[dt,dt2],[ds,ds2]])
        direction_sines=abs(np.linalg.det(directions))/np.prod(np.linalg.norm(directions,axis=0))
        tangents=D@directions
        tangent_sine=np.sqrt(max(0.,1-abs(np.vdot(tangents[:,0],tangents[:,1]))**2/np.prod(np.linalg.norm(tangents,axis=0))**2))
        diagnostics=dict(chart.F.metadata,
                         algebraic_flex_equation=residual(chart.x),
                         complex_parameter_directions=cj(directions),
                         complex_direction_independence_sine=float(direction_sines),
                         ambient_complex_tangent_independence_sine=float(tangent_sine),
                         minimum_puncture_separation=float(separations(source).min()),
                         W_basis_fit_relative_error=fit,
                         marked_parameter_jacobian=cj(D),
                         marked_parameter_jacobian_refinement_error=float(abs(D-Dc).max()),
                         marked_parameter_jacobian_singular_values=np.linalg.svd(D,compute_uv=False).tolist())
        targets={}
        if include_identity:
            targets['identity']=TargetCase(label+'_identity',source,chart.u.copy(),'identity',0.,
                                          ReferenceWarp(source.x,source.x))
        for direction,increment in (('',np.array([dt,ds])),('b',np.array([dt2,ds2]))):
          for amp in ((1.,2.,4.) if direction=='' else (1.,2.)):
            u=chart.u+amp*increment
            x=chart.at(u)
            F=Flex(*u)
            Dt=marked_jacobian(chart,u)
            conormal=np.cross(Dt[:,0],Dt[:,1])
            conormal/=np.linalg.norm(conormal)
            diag=dict(hessian_residual=F.metadata['hessian_residual'],
                      satellite_collinearity_error=F.metadata['satellite_collinearity_error'],
                      source_target_coordinate_distance=float(np.linalg.norm(x-source.x)),
                      target_flex_jacobian=cj(Dt),target_euclidean_unit_conormal=cj(conormal),
                      algebraic_flex_equation=residual(x))
            name=f'flex_{direction}{int(amp)}'
            warp_kwargs={} if direction=='' else dict(inner_fraction=.12,outer_fraction=.46)
            if amp==4:
                warp_kwargs=dict(inner_fraction=.08,outer_fraction=.48)
            targets[name]=TargetCase(label+'_'+name,Ambient(x),u,'flex',amp,ReferenceWarp(source.x,x,**warp_kwargs),diag)
            xc=x.copy()
            # The paired target has exactly the same tau and first mark.
            delta=amp*control_delta
            xc[2]+=delta
            dx=np.array([0j,0j,delta])
            diagc=dict(diag,
                       control_coordinate_displacement=cj(dx),
                       target_conormal_pairing=cj(conormal@dx),
                       control_transverse_fraction=float(abs(conormal@dx)/np.linalg.norm(dx)),
                       fixed_tau_and_w1=True,
                       dw1_ds_at_paired_flex_point=cj(Dt[1,1]),
                       source_target_coordinate_distance=float(np.linalg.norm(xc-source.x)))
            diagc['algebraic_flex_equation']=residual(xc)
            diagc['paired_flex_hessian_residual']=diagc.pop('hessian_residual')
            diagc['paired_flex_satellite_collinearity_error']=diagc.pop('satellite_collinearity_error')
            name=f'control_{direction}{int(amp)}'
            targets[name]=TargetCase(label+'_'+name,Ambient(xc),u,'transverse_control',amp,
                                    ReferenceWarp(source.x,xc,**warp_kwargs),diagc)
        cases[label]=BaseCase(label,chart,source,C,targets,diagnostics)
    return cases


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(Path(__file__).with_name('geometry_manifest.json')))
    args=parser.parse_args()
    cases=make_cases()
    manifest={name:case.metadata() for name,case in cases.items()}
    Path(args.out).write_text(json.dumps(manifest,indent=2)+'\n')
    for name,case in manifest.items():
        print(name,'source separation',case['diagnostics']['minimum_puncture_separation'],flush=True)
        for k,t in case['targets'].items():
            print(name,k,'min det',t['warp']['min_sampled_jacobian_determinant'],
                  'transverse',t['diagnostics'].get('control_transverse_fraction'),flush=True)
