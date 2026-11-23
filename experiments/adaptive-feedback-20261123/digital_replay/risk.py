"""Optimize a public feedback-to-action table under an average rejection budget."""
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

from replay import HERE, OUT, digest, matrix, read, record, table


def joint(p):
    # Row = delivered (phase, symbol), columns = true SNR. Each row has mass 1/8.
    rows = []
    for age in (2,3):
        stay = p**age
        for s in range(4):
            law = np.full(4,(1-stay)/4)
            law[s] += stay
            rows.append(law/8)
    return np.stack(rows)


def run():
    out = OUT / "risk"
    identity = {n:digest(HERE/n) for n in ["risk_protocol.json","risk.py","replay.py"]}
    record(out/"preregistration.json",{"source_identity":identity,
        "unconstrained_results_sha256":digest(OUT/"summary.csv"),"prior_split_sha256":digest(OUT/"split.json")})
    ids, candidates, val = matrix(out)
    assert read(out/"matrix_identity.json") == read(OUT/"matrix_identity.json")
    split = read(OUT/"split.json")
    train, test = ([ids.index(x) for x in split[k]] for k in ["calibration","evaluation"])
    calibration = val[train].mean(axis=0)
    controls = list(csv.DictReader((OUT/"policies.csv").open()))
    matrices, fitted = {}, []
    for p in (.60,.95):
        j = joint(p)
        quality, loss = (j @ calibration[:,:,m] for m in (0,1))
        for cap in (.01,.05):
            n = len(candidates)
            fit = linprog(-quality.ravel(),A_ub=loss.reshape(1,-1),b_ub=[cap],
                         A_eq=np.kron(np.eye(8),np.ones(n)),b_eq=np.ones(8),bounds=(0,1),method="highs")
            if not fit.success:
                raise RuntimeError(fit.message)
            mix = fit.x.reshape(8,n).clip(0,1)
            assert np.allclose(mix.sum(axis=1),1) and (mix*loss).sum() <= cap+1e-9
            matrices[p,cap] = mix
            for r,a in np.argwhere(mix>0):
                fitted.append(dict(p_design=p,cap=cap,age_slots=2+r//4,report=r%4,action=candidates[a],
                                   probability=float(mix[r,a]),training_global_risk=float((mix*loss).sum())))
        for name in ("static","two_every_two"):
            mix = np.zeros((8,len(candidates)))
            for row in (r for r in controls if float(r["p"])==p and r["mode"]==name):
                a = candidates.index(row["action"])
                if name=="static":
                    mix[:,a] = 1.
                else:
                    mix[int(row["phase"])*4+int(row["report"]),a] = 1.
            assert np.all(mix.sum(axis=1)==1.)
            matrices[p,name] = mix
    table(out/"frozen_mixtures.csv",fitted)
    record(out/"policy_freeze.json",{"calibration_sources":len(train),
           "frozen_sha256":digest(out/"frozen_mixtures.csv"),"controls_sha256":digest(OUT/"policies.csv")})
    draws = np.random.default_rng(2026112381).integers(len(test),size=(5000,len(test)))
    stats, paired = [], []
    for p in (.60,.95):
        for actual in (.60,.95):
            bas = {k:np.einsum("rs,ra,isam->im",joint(actual),matrices[p,k],val[test],optimize=True)
                   for k in (.01,.05,"static","two_every_two")}
            for cap in (.01,.05):
                for i,metric in enumerate(("dino_cosine","rejected_or_null_fraction")):
                    v=bas[cap][:,i]
                    lo,hi=np.quantile(v[draws].mean(axis=1),[.025,.975])
                    base=dict(p_design=p,p_actual=actual,cap=cap,metric=metric)
                    stats.append(dict(**base,mean=v.mean(),ci_low=lo,ci_high=hi))
                    for ctl in ("static","two_every_two"):
                        diff=v-bas[ctl][:,i]
                        lo,hi=np.quantile(diff[draws].mean(axis=1),[.025,.975])
                        paired.append(dict(**base,reference=ctl,mean_delta=diff.mean(),ci_low=lo,ci_high=hi))
    table(out/"summary.csv",stats);table(out/"paired.csv",paired)
    record(out/"completion.json",{"tables_frozen":4,"means":len(stats),"paired_differences":len(paired),
           "new_frames":0,"new_image_scores":0})


if __name__=="__main__":
    run()
