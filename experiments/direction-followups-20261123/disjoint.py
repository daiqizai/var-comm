"""Sensitivity of the 4B intervals to source images shared by cyclic user tuples."""
import csv
import numpy as np
from common import PRIMARY, PRIOR, SIC_ARMS, STUDY, digest, read, record, table


def main():
    root=STUDY/"noma"
    assert read(root/"wait.json")["actual_wait"]==0
    rows=list(csv.DictReader((root/"source_means.csv").open()))
    pairs,selection=[],[]
    for split in ("common500","kodak24"):
        ids=[r["source_id"] for r in read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))]
        n=len(ids)
        selected=ids[:n//3]
        used=set()
        for i,anchor in enumerate(selected):
            members=(anchor,ids[(i+n//3)%n],ids[(i+2*n//3)%n])
            assert len(set(members))==3 and not used.intersection(members)
            used.update(members)
            selection.append(dict(split=split,anchor=anchor,strong=members[0],middle=members[1],weak=members[2]))
        data={(r["condition"],r["arm"],r["unit"]):np.array([float(r[k]) for k in PRIMARY]) for r in rows if r["split"]==split}
        draws=np.random.default_rng(2026112308).integers(0,len(selected),(5000,len(selected)))
        for role in ("User3_4dB","all_users"):
            a={s:np.asarray([data[role,s,i] for i in selected]) for s in SIC_ARMS}
            contrasts={"both_minus_covariance":a["both"]-a["covariance"],"cancel_minus_neither":a["cancel"]-a["neither"],
                       "both_minus_cancel":a["both"]-a["cancel"],"covariance_minus_neither":a["covariance"]-a["neither"],
                       "interaction":a["both"]-a["cancel"]-a["covariance"]+a["neither"]}
            for label,d in contrasts.items():
                ci=np.quantile(d[draws].mean(axis=1),[.025,.975],axis=0)
                for m,k in enumerate(PRIMARY):
                    pairs.append(dict(split=split,condition=role,contrast=label,metric=k,independent_image_tuples=len(selected),
                                mean=float(d[:,m].mean()),ci95_lo=float(ci[0,m]),ci95_hi=float(ci[1,m])))
    table(root/"disjoint_tuples.csv",selection)
    table(root/"disjoint_paired.csv",pairs)
    record(root/"disjoint_identity.json",{"source_means_sha256":digest(root/"source_means.csv"),
        "registered_after_full_cohort_result":"Secondary analytic sensitivity; first floor(n/3) original anchors, chosen only using the cyclic role index formula. All images disjoint across retained triples. No new channel events, fitting or tuning.",
        "noise":"Original three repetitions averaged","seed":2026112308,"draws":5000,
        "limitations":"Eight Kodak tuples. This check does not vary the identity of a given source's two interferers."})


if __name__=="__main__":
    main()
