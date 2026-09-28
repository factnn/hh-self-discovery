"""Counterfactual output sensitivity to replacing dynamic m by m_inf(V)."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
from hh_self_discovery.simulator import ionic_currents

def m_inf(voltage):
    x=voltage+40.0; ratio=x/10.0
    trap=np.where(np.abs(ratio)<1e-7,10.0*(1.0+ratio/2.0+ratio*ratio/12.0),x/(-np.expm1(-ratio)))
    alpha=0.1*trap; beta=4.0*np.exp(np.clip(-(voltage+65.0)/18.0,-700,700))
    return alpha/(alpha+beta)

def summarize(records):
    true=np.concatenate([r['true'] for r in records]); cf=np.concatenate([r['cf'] for r in records]); ina=np.concatenate([r['ina'] for r in records]); active=np.concatenate([r['active'] for r in records])
    delta=cf-true; out={'points':int(len(true)),'trajectories':len(records),'current_r2':float(r2_score(true,cf)),'delta_rmse':float(np.sqrt(np.mean(delta**2))),'delta_rmse_over_current_std':float(np.sqrt(np.mean(delta**2))/max(np.std(true),1e-12)),'delta_rmse_over_sodium_rms':float(np.sqrt(np.mean(delta**2))/max(np.sqrt(np.mean(ina**2)),1e-12))}
    if active.any(): out['active_delta_rmse_over_current_std']=float(np.sqrt(np.mean(delta[active]**2))/max(np.std(true[active]),1e-12)); out['active_fraction']=float(active.mean())
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--split',default='test'); p.add_argument('--output',required=True); p.add_argument('--max-trajectories',type=int,default=0); a=p.parse_args()
    root=Path(a.data); manifest=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if json.loads(x)['split']==a.split]
    if a.max_trajectories: manifest=manifest[:a.max_trajectories]
    groups={}
    for row in manifest:
        obs=np.load(root/row['observed_path']); ev=np.load(root/row['evaluation_path']); v=obs['voltage_mV']; n,m,h=ev['n'],ev['m'],ev['h']; ina,ik,il=ionic_currents(v,n,m,h); cina,cik,cil=ionic_currents(v,n,m_inf(v),h)
        record={'true':ina+ik+il,'cf':cina+cik+cil,'ina':ina,'active':obs['event_label']>0}; groups.setdefault(row['control_mode'],[]).append(record)
    result={mode:summarize(records) for mode,records in groups.items()}; result['all']=summarize(sum(groups.values(),[])); Path(a.output).write_text(json.dumps({'split':a.split,'results':result},indent=2,sort_keys=True)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
