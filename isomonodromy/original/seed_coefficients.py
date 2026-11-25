"""Fix spectral-line input coefficients before any metric or transport solve."""
import json
from pathlib import Path
import numpy as np
import flex_wp as fw
from seed_higgs import BASES,cutoff,cj

def compute(base,N):
    F=fw.Flex(*BASES[base]);fw.quadrature_order(6)
    M=fw.mesh(F,N,1e-4,64,1.18);z=M['tz']@fw.BARY.T
    psi=F.differentials(z);chi,_=cutoff(F,z,M['sep']);w=M['area'][:,None]*fw.BW
    G=np.einsum('tq,tqi,tqj->ij',w*chi,psi,psi.conj())
    norms=np.sqrt(G.diagonal().real);c=G[2,0]/G[2,2]
    q=(psi[...,0]/norms[0]+(.4+.3j)*psi[...,1]/norms[1])/4
    f=chi*(psi[...,0].conj()-c*psi[...,2].conj())
    scale=1/np.sqrt(2*np.sum(w*abs(q)*abs(f)**2))
    return dict(base=base,N=N,norms=norms.tolist(),kernel_correction=cj(c),f_scaling=float(scale),serre_gram=cj(G),
                input_quadrature=6,input_cutoff=1e-4,parameters=cj(BASES[base]))

if __name__=='__main__':
    p=Path(__file__).parent/'inputs';p.mkdir(exist_ok=True)
    for base in BASES:
        a=compute(base,64);b=compute(base,96)
        b['coefficient_change_from_N64']={k:float(np.linalg.norm(np.array(a[k])-np.array(b[k]))) for k in ['norms','kernel_correction','f_scaling']}
        (p/f'{base}_coefficients.json').write_text(json.dumps(b,indent=2)+'\n');print(base,b['coefficient_change_from_N64'],flush=True)
