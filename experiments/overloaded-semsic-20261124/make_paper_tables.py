"""Format the two paper tables directly from full-precision paired outputs."""
import argparse
import csv
import json
from pathlib import Path


def keyed(path, label):
    with path.open(newline="") as f:
        return {(r["condition"],r[label],r["metric"]):r for r in csv.DictReader(f) if r["split"]=="common500"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("analysis",type=Path)
    p.add_argument("paper",type=Path)
    a=p.parse_args()
    summary=keyed(a.analysis / "summary.csv","arm")
    fixed=keyed(a.analysis / "gate_paired.csv","comparison")
    levels=("nearfar_14db","nearfar_18db_ood")
    methods=[("linear_fixed_gate","Linear, fixed gates"),("latent","Denoised latent"),("rgb_reencode","RGB re-encoding")]
    def fmt(x,d):
        return f"{float(x):.{d}f}".replace("-","−")
    rows=[]
    for arm, display in methods:
        row=[display]
        for level in levels:
            for metric,d in (("weak_psnr_db",3),("weak_dists",4),("interference_nmse_db_r3",3)):
                row.append(fmt(summary[level,arm,metric]["mean"],d))
        rows.append(row)
    follow=[]
    for metric,label,d in [("weak_psnr_db","Weak PSNR (dB)",3),("weak_dists","Weak DISTS",4),
                           ("interference_nmse_db_r3","Interference NMSE (dB)",3)]:
        row=[label]
        for level in levels:
            x=fixed[level,"rgb_fixed_minus_latent",metric]
            row.append(f'{fmt(x["mean"],d)}  [{fmt(x["ci_low"],d)}, {fmt(x["ci_high"],d)}]')
        follow.append(row)
    a.paper.mkdir(exist_ok=True,parents=True)
    (a.paper / "table_cells.json").write_text(json.dumps(dict(absolute=rows,fixed=follow),indent=2,ensure_ascii=False),encoding="utf-8")


if __name__ == "__main__":
    main()
