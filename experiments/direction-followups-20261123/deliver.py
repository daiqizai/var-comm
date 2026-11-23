"""Assemble the lightweight Git payload from verified, ordinary files on PFS."""
from pathlib import Path
from common import CKPT, PRIOR, STUDY, TEMP, digest, path_check, read, record, save

ED = Path("experiments/direction-followups-20261123")
RD = Path("reports/direction_followups_20261123")


def main():
    qa=read(STUDY/"data_checks.json")
    assert set(qa)=={"noma","pic","wave_test","rf","feedback","qam"}
    root_files=["README_zh.md","protocol.json","wave_inference_protocol.json","causal_checks.json",
                "data_checks.json","storage_audit.json","resource_ledger.json"]
    per_track={
        "noma":["disjoint_tuples.csv","disjoint_paired.csv","disjoint_identity.json"],
        "pic":["protocol.json","frozen_damping.json","validation_sources.csv"],
        "qam":["protocol.json","frozen_gains.json","preflight.json","channel_check.json","test_source_power.csv",
               "io_recovery.json","start.json","wait.json","recovery_start.json","recovery_wait.json"],
        "wave_test":[],
        "feedback":["protocol.json","estimator.json","history_estimates.csv"],
        "rf":["protocol.json","calibration.json","calibration_sources.json","cpu_placement_revision.json"],
    }
    for track,extra in per_track.items():
        for name in ["means.csv","paired.csv","source_means.csv","evaluation.json","statistics_identity.json",*extra]:
            root_files.append(track+"/"+name)
    root_files.append("qualify_cache/identity.json")
    for tag in ("spread","no_spread"):
        for f in ("initialization.json","summary_"+tag+".json","history_"+tag+".csv"):
            root_files.append("training/"+tag+"/"+f)
    context=read(TEMP/"main_verification/context.json")
    view=Path(context["view"])
    assert read(TEMP/"main_verification/checkout_exit.json")["actual_exit"]==0
    files={str(ED/p.name):p for p in STUDY.glob("*.py")}
    files[str(ED/"README.md")]=STUDY/"README.md"
    files.update({str(RD/n):STUDY/n for n in root_files})
    listing={}
    for rel,p in sorted(files.items()):
        src=path_check(p)
        sha=digest(src)
        assert src.stat().st_size < 10_000_000
        save(view/rel,src.read_bytes())
        assert digest(view/rel)==sha
        listing[rel]={"sha256":sha,"bytes":src.stat().st_size}
    raw={}
    for track in per_track:
        for split in ("common500","kodak24"):
            p=STUDY/track/(split+"_frames.csv")
            raw[str(p)]={"bytes":p.stat().st_size,"sha256":digest(p)}
    weights={str(p):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(CKPT.glob("*/*.pt"))}
    assert len(weights)==4
    manifest={"parent":context["parent"],"included":listing,"full_event_scores_on_pfs":raw,
        "new_checkpoints_on_pfs":weights,"study_root":str(STUDY),"runtime_root":str(TEMP),
        "source_identity":{s:digest(PRIOR/"evaluation/noma"/(s+"_source_manifest.json")) for s in ("common500","kodak24")}}
    record(STUDY/"artifact_manifest.json",manifest)
    save(view/RD/"artifact_manifest.json",(STUDY/"artifact_manifest.json").read_bytes())
    print(len(listing)+1,"lightweight files",len(raw),"event files",len(weights),"checkpoint files")


if __name__=="__main__":
    main()
