"""Fit a globally stable and bounded relaxation law to decoded gates."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from audit_relaxation_structure import chart, design, unpack
from hh_self_discovery.blind_evaluation import TRUE_GATES, _collect_windows, load_latent_checkpoint
from hh_self_discovery.simulator import gate_rates


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--blind-result',required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--windows-per-mode',type=int,default=200); p.add_argument('--seed',type=int,default=12800); p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args(); device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model,ckpt=load_latent_checkpoint(a.checkpoint,device); root=Path(a.data)
    mapping=_collect_windows(model,ckpt,root,'validation',device,a.windows_per_mode,a.seed,stride=1)
    test=_collect_windows(model,ckpt,root,'test',device,a.windows_per_mode,a.seed+10000,stride=1)
    blind=json.load(open(a.blind_result)); readout=chart(blind['ridge_alpha'])
    readout.fit(np.concatenate([x['latents'] for x in mapping]),np.concatenate([x['gates'] for x in mapping]))
    nsteps=ckpt['prediction_steps']; train=unpack(mapping,readout,nsteps,ckpt['dt_ms']); evaluation=unpack(test,readout,nsteps,ckpt['dt_ms'])
    spline=SplineTransformer(n_knots=12,degree=3,include_bias=True).fit(train[0][:,None])
    bt=spline.transform(train[0][:,None]).astype('float32'); be=spline.transform(evaluation[0][:,None]).astype('float32')
    voltage_grid=np.linspace(-100.0,50.0,301); bg=spline.transform(voltage_grid[:,None]).astype('float32')
    true_steady=[]; true_tau=[]
    for voltage in voltage_grid:
        an,bn,am,bm,ah,bh=gate_rates(float(voltage)); alpha=np.asarray((an,am,ah)); beta=np.asarray((bn,bm,bh))
        true_steady.append(alpha/(alpha+beta)); true_tau.append(1.0/(alpha+beta))
    true_steady=np.asarray(true_steady); true_tau=np.asarray(true_tau)
    physics_voltage=np.linspace(-200.0,100.0,6001); physics_steady=[]; physics_tau=[]
    for voltage in physics_voltage:
        an,bn,am,bm,ah,bh=gate_rates(float(voltage)); alpha=np.asarray((an,am,ah)); beta=np.asarray((bn,bm,bh))
        physics_steady.append(alpha/(alpha+beta)); physics_tau.append(1.0/(alpha+beta))
    physics_steady=np.asarray(physics_steady); physics_tau=np.asarray(physics_tau)
    eval_true_steady=np.column_stack([np.interp(evaluation[0],physics_voltage,physics_steady[:,g]) for g in range(3)])
    eval_true_tau=np.column_stack([np.interp(evaluation[0],physics_voltage,physics_tau[:,g]) for g in range(3)])
    rng=np.random.default_rng(a.seed); result={}
    for gi,gate in enumerate(TRUE_GATES):
        unconstrained=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-4,4,9)))
        unconstrained.fit(design(bt,train[2][:,gi],1),train[4][:,gi])
        unconstrained_r2=float(r2_score(evaluation[4][:,gi],unconstrained.predict(design(be,evaluation[2][:,gi],1))))
        coefficients=torch.zeros((2,bt.shape[1]),device=device,requires_grad=True)
        optimizer=torch.optim.Adam([coefficients],lr=0.03,weight_decay=1e-6)
        scale=max(float(np.std(train[4][:,gi])),1e-6)
        for _ in range(a.steps):
            idx=rng.integers(0,len(bt),size=min(8192,len(bt)))
            basis=torch.from_numpy(bt[idx]).to(device); state=torch.from_numpy(train[2][idx,gi].astype('float32')).to(device); target=torch.from_numpy(train[4][idx,gi].astype('float32')).to(device)
            steady=torch.sigmoid(basis@coefficients[0]); tau=torch.nn.functional.softplus(basis@coefficients[1])+0.02
            loss=torch.mean(((steady-state)/tau-target)**2)/(scale*scale); optimizer.zero_grad(); loss.backward(); optimizer.step()
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(be),65536):
                basis=torch.from_numpy(be[start:start+65536]).to(device); state=torch.from_numpy(evaluation[2][start:start+65536,gi].astype('float32')).to(device)
                predictions.append(((torch.sigmoid(basis@coefficients[0])-state)/(torch.nn.functional.softplus(basis@coefficients[1])+0.02)).cpu().numpy())
        pred=np.concatenate(predictions); constrained_r2=float(r2_score(evaluation[4][:,gi],pred))
        with torch.no_grad():
            grid=torch.from_numpy(bg).to(device)
            learned_steady=torch.sigmoid(grid@coefficients[0]).cpu().numpy()
            learned_tau=(torch.nn.functional.softplus(grid@coefficients[1])+0.02).cpu().numpy()
        log_tau_error=np.log(learned_tau)-np.log(true_tau[:,gi]); log_tau_bias=float(np.mean(log_tau_error))
        true_state_predictions=[]; oracle_clock_predictions=[]; learned_steady_parts=[]; learned_tau_parts=[]
        with torch.no_grad():
            for start in range(0,len(be),65536):
                basis=torch.from_numpy(be[start:start+65536]).to(device)
                state=torch.from_numpy(evaluation[1][start:start+65536,gi].astype('float32')).to(device)
                steady=torch.sigmoid(basis@coefficients[0]); tau=torch.nn.functional.softplus(basis@coefficients[1])+0.02
                learned_steady_parts.append(steady.cpu().numpy()); learned_tau_parts.append(tau.cpu().numpy())
                true_state_predictions.append(((steady-state)/tau).cpu().numpy())
                oracle_clock_predictions.append(((steady-state)/(tau/np.exp(log_tau_bias))).cpu().numpy())
        true_state_predictions=np.concatenate(true_state_predictions); oracle_clock_predictions=np.concatenate(oracle_clock_predictions)
        eval_learned_steady=np.concatenate(learned_steady_parts); eval_learned_tau=np.concatenate(learned_tau_parts)
        field_true_steady=(eval_true_steady[:,gi]-evaluation[1][:,gi])/eval_learned_tau
        field_true_tau=(eval_learned_steady-evaluation[1][:,gi])/eval_true_tau[:,gi]
        voltage_for_correction=np.clip(evaluation[0],voltage_grid[0],voltage_grid[-1]); tau_calibration={}
        log_ratio=np.log(true_tau[:,gi])-np.log(learned_tau)
        for degree in range(6):
            correction=np.polynomial.Polynomial.fit(voltage_grid,log_ratio,degree)
            corrected_tau=eval_learned_tau*np.exp(correction(voltage_for_correction))
            corrected_field=(eval_learned_steady-evaluation[1][:,gi])/corrected_tau
            tau_calibration[f'degree_{degree}']=float(r2_score(evaluation[3][:,gi],corrected_field))
        result[gate]={'constrained_test_r2':constrained_r2,'unconstrained_affine_test_r2':unconstrained_r2,'constraint_r2_regret':unconstrained_r2-constrained_r2,
                      'steady_spearman':float(spearmanr(learned_steady,true_steady[:,gi]).statistic),
                      'tau_log_spearman':float(spearmanr(np.log(learned_tau),np.log(true_tau[:,gi])).statistic),
                      'tau_log_rmse':float(np.sqrt(np.mean(log_tau_error**2))),
                      'tau_log_bias':log_tau_bias,
                      'tau_geometric_scale':float(np.exp(log_tau_bias)),
                      'tau_centered_log_rmse':float(np.sqrt(np.mean((log_tau_error-log_tau_bias)**2))),
                      'voltage_grid_mV':voltage_grid.tolist(),
                      'learned_tau_ms':learned_tau.tolist(),
                      'true_tau_ms':true_tau[:,gi].tolist(),
                      'analytic_field_r2_on_true_state':float(r2_score(evaluation[3][:,gi],true_state_predictions)),
                      'oracle_clock_corrected_analytic_field_r2':float(r2_score(evaluation[3][:,gi],oracle_clock_predictions)),
                      'true_steady_learned_tau_field_r2':float(r2_score(evaluation[3][:,gi],field_true_steady)),
                      'learned_steady_true_tau_field_r2':float(r2_score(evaluation[3][:,gi],field_true_tau)),
                      'oracle_voltage_tau_calibration_field_r2':tau_calibration}
    Path(a.output).write_text(json.dumps({'checkpoint':a.checkpoint,'results':result},indent=2,sort_keys=True)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
