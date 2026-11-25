"""High-resolution forward map, followed by an unconstrained flatness audit."""
import argparse,json
from pathlib import Path
import numpy as np
from seed_higgs import Seed
from connection_check import edge_data
from flat_bundle import reconstruct

ROOT=Path(__file__).parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='A');ap.add_argument('--control',action='store_true');ap.add_argument('--N',type=int,default=64);a=ap.parse_args()
    name=f'{a.base}_{"control" if a.control else "fibre"}_N{a.N}';out=ROOT/'canonical'/name;out.mkdir(parents=True,exist_ok=True)
    o=Seed(a.base,N=a.N,R=.0000625,ntheta=80,ratio=1.18,control=a.control,coeff_path=ROOT/'inputs'/f'{a.base}_coefficients.json')
    seed=o.solve();(out/'seed.json').write_text(json.dumps(seed,indent=2)+'\n')
    data=edge_data(o,steps=12)
    np.savez_compressed(out/'raw_connection.npz',**{k:data[k] for k in ['edges','links','tri','tz','z','free','node_U']},s=o.s,u=o.u)
    links,audit,extra=reconstruct(data,o.F)
    (out/'flat_projection.json').write_text(json.dumps(audit,indent=2)+'\n')
    np.savez_compressed(out/'flat_connection.npz',**{k:data[k] for k in ['edges','tri','tz','z','free','node_U']},links=links,
                        frames=extra['frames'],generators=extra['generators'])
    print(json.dumps(audit,indent=2),flush=True)
