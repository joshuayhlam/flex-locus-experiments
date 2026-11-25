"""Independent coordinate, tensor-convention and saved-solve checks."""
import json
from pathlib import Path
import numpy as np
from second_fundamental import FlexChart,Ambient,levi_civita,uncj,cj,second_form

records=[]
for label,tau,s in [('A',.23+1.08j,5+2j),('B',.07+.94j,-4+3j),('C',.31+.83j,-2+7j)]:
    C=FlexChart(tau,s);A=Ambient(C.x);D,DD=C.jets();D2,DD2=C.jets(.7,40)
    z=np.array([.13+.28j,.52+.38j,.72+.21j])
    fit=np.linalg.solve(A.differentials(z),C.F.differentials(z+C.ref[0])[:,2])
    conormal=fit@A.duality()
    vmax=A.T.uv(C.x[1:])[:,1].max();base=-(1-vmax)/2*tau-.03
    per=A.differentials(base+(np.arange(4096)+.5)/4096).mean(axis=0)
    record=dict(sample=label,period_pairing_error=float(np.max(abs(.5j*per-A.duality()[:,0]))),
                conormal_relative_error=float(np.linalg.norm(conormal@D)/np.linalg.norm(conormal)/np.linalg.norm(D)),
                first_jet_check=float(np.max(abs(D-D2))),second_jet_check=float(np.max(abs(DD-DD2))))
    records.append(record)

# Product metric exp(2 Re z0)|dz0|^2+|dz1|^2+|dz2|^2:
# Gamma^x_xx=1, Gamma^x_yy=-1, Gamma^y_xy=Gamma^y_yx=1.
g=np.eye(3,dtype=complex);dg=np.zeros((6,3,3),complex);dg[0,0,0]=2
LC=levi_civita(g,dg);expected=np.zeros((6,6,6))
expected[0,0,0]=1;expected[0,3,3]=-1;expected[3,0,3]=expected[3,3,0]=1
lc_error=float(np.max(abs(LC-expected)))
assert lc_error<1e-14

metrics=[]
for p in Path('results').glob('*/metrics/*.json'):
    a=json.loads(p.read_text());G=uncj(a['gram'])
    ev=np.linalg.eigvalsh(G)
    assert ev.min()>0
    assert a['newton_residual']<2e-9
    metrics.append((a['newton_residual'],abs(a['gauss_bonnet_error'])))
answer=dict(coordinate_audits=records,real_connection_product_metric_error=lc_error,
            successful_metric_records=len(metrics),max_newton_residual=max(x[0] for x in metrics),
            max_gauss_bonnet_bookkeeping_error=max(x[1] for x in metrics))
Path('results/audit.json').write_text(json.dumps(answer,indent=2))
print(json.dumps(answer,indent=2))
