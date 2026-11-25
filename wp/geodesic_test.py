"""Ambient real Levi-Civita geodesic shooting, with no flex projection.

Third-order Runge--Kutta; flex/bent-surface membership is evaluated only
after integration. The metric and its derivatives are solved off the locus.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from second_fundamental import Metric,FlexChart,cj,uncj,real_metric


def asreal(z):return np.r_[z.real,z.imag]
def ascomplex(z):return z[:3]+1j*z[3:]


def main(args):
    source=json.loads(Path(args.source).read_text());cfg=source['metric_config'].copy()
    cfg.pop('degree');cfg.pop('background')
    metric=Metric(Path(args.out)/'metrics',**cfg)
    chart=FlexChart(complex(*source['tau']),complex(*source['s']))
    D=uncj(source['D']);E=uncj(source['orthonormal_parameter_frame'])
    direction=np.array([1,1j])/np.sqrt(2)
    pu=E@direction;v=D@pu
    normal=uncj(source['unit_normal']);c=uncj(source['unit_conormal'])
    target=uncj(source['control_prescribed_B_increment'])
    bend=np.linalg.inv(E).T@target@np.linalg.inv(E)
    x0=asreal(chart.x);v0=asreal(v)

    def rhs(y):
        x=ascomplex(y[:6]);vel=y[6:]
        metric.connection(x,args.h)
        acc=-np.einsum('kij,i,j->k',metric.real_connection,vel,vel)
        return np.r_[vel,acc]

    def rk3(y,dt):
        k1=rhs(y);k2=rhs(y+.5*dt*k1);k3=rhs(y-dt*k1+2*dt*k2)
        return y+dt*(k1+4*k2+k3)/6

    def match(x,t,bent):
        def surface(u):
            z=chart.at(u)
            if bent:
                du=u-chart.u;z=z+.5*normal*(du@bend@du)
            return z
        u=chart.u+t*pu
        for it in range(10):
            z=surface(u);res=z[:2]-x[:2]
            if np.linalg.norm(res)<2e-13:break
            J=[]
            for k in range(2):
                d=np.zeros(2,complex);d[k]=1e-4
                J.append((surface(u+d)[:2]-surface(u-d)[:2])/(2e-4))
            u-=np.linalg.solve(np.array(J).T,res)
        z=surface(u);return c@(x-z),float(np.linalg.norm(z[:2]-x[:2]))

    results=[]
    for t in args.times:
        for steps in args.steps:
            start=time.time();y=np.r_[x0,v0]
            for k in range(steps):y=rk3(y,t/steps)
            x=ascomplex(y[:6]);res,err=match(x,t,False);ctl,cerr=match(x,t,True)
            g=metric(x);speed=np.sqrt(np.real(ascomplex(y[6:]).conj()@g@ascomplex(y[6:])))
            record=dict(time=t,steps=steps,endpoint=cj(x),velocity=cj(ascomplex(y[6:])),
                normal_residual=cj(res),normal_residual_over_t2=cj(res/t**2),
                control_normal_residual=cj(ctl),control_normal_residual_over_t2=cj(ctl/t**2),
                matching_error=err,control_matching_error=cerr,speed=float(speed),seconds=time.time()-start)
            results.append(record)
            Path(args.out).mkdir(parents=True,exist_ok=True)
            (Path(args.out)/'geodesics.json').write_text(json.dumps(dict(source=args.source,
                metric_config=cfg,h=args.h,direction=cj(direction),control_predicted_residual_over_t2=cj(-.5*direction@target@direction),results=results),indent=2))
            print('GEODESIC',json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--h',type=float,default=.0005);p.add_argument('--times',nargs='+',type=float,default=[.04,.02])
    p.add_argument('--steps',nargs='+',type=int,default=[1,2]);main(p.parse_args())
