"""Locate where dynamic-m information occurs in existing trajectories."""
import argparse, json
from pathlib import Path
import numpy as np
from audit_m_instantaneous_observability import m_inf
from hh_self_discovery.simulator import ionic_currents

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--split',default='test'); p.add_argument('--output',required=True); a=p.parse_args(); root=Path(a.data)
    rows=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines()]; rows=[x for x in rows if x['split']==a.split]; groups={}
    for row in rows:
        obs=np.load(root/row['observed_path']); ev=np.load(root/row['evaluation_path']); v=obs['voltage_mV']; n,m,h=ev['n'],ev['m'],ev['h']; ina,ik,il=ionic_currents(v,n,m,h); cina,_,_=ionic_currents(v,n,m_inf(v),h); delta=cina-ina; dt=float(row['dt_ms'])
        dv=np.abs(np.diff(v,prepend=v[0])); jump=np.flatnonzero(dv>1.0); since=np.full(len(v),np.inf)
        last=-10**9
        jump_set=set(jump.tolist())
        for i in range(len(v)):
            if i in jump_set: last=i
            since[i]=(i-last)*dt
        groups.setdefault(row['control_mode'],[]).append((delta,dv,since))
    result={}
    for mode,items in groups.items():
        delta=np.concatenate([x[0] for x in items]); dv=np.concatenate([x[1] for x in items]); since=np.concatenate([x[2] for x in items]); energy=delta**2; total=max(float(energy.sum()),1e-30); order=np.sort(energy)[::-1]
        concentration={f'top_{pct}pct':float(order[:max(1,int(len(order)*pct/100))].sum()/total) for pct in (0.1,0.5,1,5,10)}
        transition={}
        for ms in (0.1,0.2,0.5,1.0):
            mask=since<=ms; transition[f'{ms:g}_ms']={'sample_fraction':float(mask.mean()),'energy_fraction':float(energy[mask].sum()/total)}
        result[mode]={'points':int(len(delta)),'energy_concentration':concentration,'post_voltage_jump':transition,'abs_dv_energy_weighted_mean':float(np.sum(dv*energy)/total),'abs_dv_unweighted_mean':float(np.mean(dv))}
    Path(a.output).write_text(json.dumps({'split':a.split,'results':result},indent=2,sort_keys=True)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
