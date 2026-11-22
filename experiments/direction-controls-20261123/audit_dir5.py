"""Audit the producer's light evidence on the two test sets, distinguishing the two releases."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from study import STUDY, TEMP, dump_once, save_once


def rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def audit(label):
    snap=TEMP/"producer_snapshots"/label
    base=snap/"results/dir5_otfs_genvar_20261122/eval"
    receipt=json.loads((snap/"manifest.json").read_text())
    out=STUDY/"dir5"/label
    checked, differences, values={},[],[]
    for split,n in (("common500",500),("kodak24",24)):
        summary=json.loads((base/f"summary_dir5_{split}.json").read_text())
        assert summary["num_images"]==n and summary["noise_seeds"]==[5001,5002,5003]
        for family,main in (("dir5a_otfs","Symp_OTFS_JSCC"),("dir5b_genvar","GenHolo_VAR_JSCC")):
            means=base/f"{family}_{split}.csv";paired=base/f"{family}_paired_{split}.csv"
            for f in (means,paired):
                buf=f.read_bytes()
                assert hashlib.sha256(buf).hexdigest()==receipt["files"][str(f.relative_to(snap))]
                save_once(out/f.name,buf)
            items=rows(means)
            index={(z["condition"],z["model"]):z for z in items}
            pairs=rows(paired)
            expected=(66,55) if family.startswith("dir5a") else (48,36)
            assert len(items)==len(index)==expected[0] and len(pairs)==expected[1]
            for row in items:
                for metric in ("psnr_db","ms_ssim","lpips_alex","dists"):
                    mu,lo,hi=(float(row[metric+k]) for k in ("_mean","_ci95_lo","_ci95_hi"))
                    assert lo<=mu<=hi
            if family.startswith("dir5a"):
                a="snr_sweep_kd010_10dB";b="vel_sweep_10dB_kd010"
                for z in items:
                    if z["condition"]==a:
                        assert all(z[k]==index[(b,z["model"])][k] for k in z if "_mean" in k or "_ci95" in k)
            maxerr=0.
            for z in pairs:
                prop,ref=z["comparison"].split("_minus_")
                assert prop==main
                cond=z["condition"];one=index[cond,main];two=index[cond,ref]
                maps=[("psnr_db","psnr"),("dists","dists")]
                if family.startswith("dir5b"): maps.append(("lpips_alex","lpips"))
                for m,field in maps:
                    delta=float(z[field+"_diff_mean"])
                    residual=(float(one[m+"_mean"])-float(two[m+"_mean"]))-delta
                    maxerr=max(maxerr,abs(residual)); assert abs(residual)<1.6e-6
                    key=field+"_ci95_lo"
                    lo=z.get(key,"");hi=z.get(field+"_ci95_hi","")
                    if lo: assert float(lo)<=delta<=float(hi)
                    differences.append({"release":label,"track":family[:5],"split":split,"n_sources":n,
                            "condition":cond,"method":main,"reference":ref,"metric":m,
                            "paired_mean":delta,"ci95_lo":lo,"ci95_hi":hi})
            checked[family+"_"+split]={"mean_rows":len(items),"paired_rows":len(pairs),
                "means_check_max":maxerr,"distinct_physical_conditions":10 if family.startswith("dir5a") else 12}
    stream=io.StringIO(newline="")
    w=csv.DictWriter(stream,list(differences[0]));w.writeheader();w.writerows(differences)
    save_once(out/"all_paired_long.csv",stream.getvalue().encode())
    dump_once(out/"audit.json",{"release":label,"source":str(base),"checks":checked,
              "paired_quantities_checked":len(differences),"checkpoint_match_at_snapshot":receipt["checkpoints_matching_all"],
              "duplicate_condition":"5A: k_d=.10 at 10 dB repeated in both sweeps; same RNG and same output",
              "unavailable_CIs":"Producer provides 5A PSNR intervals and 5B DISTS/LPIPS intervals. Blanks mark unavailable intervals.",
              "scope":"Aggregate consistency and checkpoint identities. Source-level bootstraps in producer tables were not independently recomputed."})
    print(label,len(differences),"direct mean differences reconciled")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("label")
    audit(parser.parse_args().label)
