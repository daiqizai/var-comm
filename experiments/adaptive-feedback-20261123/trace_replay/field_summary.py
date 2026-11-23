"""Summarize completed session-level measurements, including empty or singleton strata."""
import csv
from itertools import combinations
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from study import ROOT, SPEC, record, table


def write_summary(out, seed):
    with (out / "sessions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (out / "coverage.csv").open(newline="") as f:
        coverage = list(csv.DictReader(f))
    summaries, contrasts, inclusion = [], [], []
    for group in ("All","Static","Driving"):
        use = [r for r in rows if group=="All" or r["mobility"]==group]
        sessions = sorted(set(r["session"] for r in use))
        inclusion.append(dict(mobility=group,qualifying_sessions=len(sessions),
                         interval_estimable=len(sessions)>1))
        if not sessions:
            continue
        idx = np.random.default_rng(seed).integers(len(sessions),size=(5000,len(sessions)))
        def interval(values):
            return np.quantile(values,[.025,.975]) if len(sessions)>1 else ("","")
        for decoder in ("synthetic_fit","last_decoded"):
            methods = list(SPEC["modes"]) + (["unquantized"] if decoder=="last_decoded" else [])
            vals = {}
            for method in methods:
                lookup = {r["session"]:float(r["mse_db2"]) for r in use if r["decoder"]==decoder and r["mode"]==method}
                score = np.array([lookup[n] for n in sessions])
                rmse = float(np.sqrt(score.mean()))
                boot = np.sqrt(score[idx].mean(axis=1))
                vals[method] = (rmse,boot)
                lo,hi = interval(boot)
                summaries.append(dict(mobility=group,decoder=decoder,mode=method,sessions=len(sessions),
                                      rmse_db=rmse,ci_low=lo,ci_high=hi))
            for a,b in combinations(methods,2):
                mean=vals[a][0]-vals[b][0]
                lo,hi=interval(vals[a][1]-vals[b][1])
                contrasts.append(dict(mobility=group,decoder=decoder,left=a,right=b,delta_rmse_db=mean,
                                      ci_low=lo,ci_high=hi))
    table(out / "summary.csv", summaries)
    table(out / "paired.csv", contrasts)
    record(out / "coverage.json",{"files":len(coverage),"inclusion":inclusion,
           "with_scored_seconds":sum(int(r["scored_seconds"])>0 for r in coverage),
           **{k:sum(int(r[k]) for r in coverage) for k in ("raw_rows","duplicates","invalid_seconds",
           "segments","qualifying_segments","scored_seconds","outside_simulated_range")}})


if __name__=="__main__":
    if sys.argv[1] not in ("field","field_clock"):
        raise ValueError("Unknown field output")
    write_summary(ROOT / sys.argv[1],2026112327)
