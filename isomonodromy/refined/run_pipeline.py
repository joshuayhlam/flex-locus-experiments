"""Run a selected forward refinement, retaining only F1 inverse tests."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent
if __name__=='__main__':
    import fcntl
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--wait-candidate',action='store_true');a=ap.parse_args()
    lock=(ROOT/f'pipeline_{a.base}_{a.stage}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (ROOT/f'pipeline_{a.base}_{a.stage}_status.json').exists():raise SystemExit(0)
    records=[]
    for control in [False,True]:
        name=a.base+('_control' if control else '_fibre')
        args=['--base',a.base,'--stage',a.stage]+(['--control'] if control else [])
        if a.wait_candidate and not control:
            while not (ROOT/'forward'/f'{name}_{a.stage}'/'flat_connection.npz').exists():time.sleep(5)
        else:
            start=time.time()
            with (ROOT/f'forward_{name}_{a.stage}.log').open('w') as f:
                p=subprocess.run([sys.executable,'refine_forward.py']+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            records.append(dict(task='forward',name=name,returncode=p.returncode,seconds=time.time()-start))
            assert p.returncode==0,records[-1]
        start=time.time()
        with (ROOT/f'target_{name}_{a.stage}_F1.log').open('w') as f:
            p=subprocess.run([sys.executable,'repeat_F1.py']+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(task='target_F1',name=name,returncode=p.returncode,seconds=time.time()-start))
        print(json.dumps(records[-1]),flush=True);assert p.returncode==0,records[-1]
    (ROOT/f'pipeline_{a.base}_{a.stage}_status.json').write_text(json.dumps(records,indent=2))
