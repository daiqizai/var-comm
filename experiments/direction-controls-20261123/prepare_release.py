"""Assemble reviewed ordinary files into an isolated validation view of the main Git tree."""
import hashlib
import json
from pathlib import Path
from study import STUDY, TEMP, regular_path, save_once, dump_once

EXPERIMENT = Path("experiments/direction-controls-20261123")
REPORT = Path("reports/directions_20261123")
SCRIPTS = (
    "study.py","qualify.py","train_pair.py","evaluate_pairs.py","after_training.py",
    "capture_dir5.py","audit_dir5.py","seal_dir5.py","audit_history.py",
    "dir5_channel_check.py","dir5_receiver_check.py","dir5_waveform_check.py",
    "finish_checks.py","results_digest.py","prepare_release.py",
    "dir5_training_pair.py","dir5_training_evaluation.py","dir5_training_audit.py",
)


def main():
    temp_view = TEMP/"main_verification"
    view = temp_view/"view"
    assert json.loads((temp_view/"checkout_exit.json").read_text())["actual_wait"]==0
    for name in ("README_zh.md","DIRECTIONS_zh.md","CONTROLS_zh.md","resource_ledger.json","data_checks.json","storage_audit.json","combined_resource.json","dir5/train_control_256/review.json"):
        assert (STUDY/name).is_file(), name
    included={}
    pfs_only={}

    def record(source,target):
        regular_path(source)
        data=source.read_bytes()
        if len(data)>10_000_000:
            raise ValueError(f"Unexpected size for Git: {source}")
        save_once(view/target,data)
        included[str(target)]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}

    for name in SCRIPTS:
        record(STUDY/name,EXPERIMENT/name)
    record(STUDY/"REPRODUCE.md",EXPERIMENT/"README.md")
    for path in sorted(STUDY.rglob("*")):
        if not path.is_file() or path.suffix not in {".md",".csv",".json"}:continue
        relative=path.relative_to(STUDY)
        if path.name=="REPRODUCE.md" or path.name in ("light_delivery_manifest.json","publication_receipt.json","final_file_check.json"):
            continue
        if path.name.endswith("_frame_scores.csv"):
            regular_path(path)
            data=path.read_bytes()
            pfs_only[str(relative)]={"path":str(path),"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),
                                     "scope":"raw repeated frame scores; source means included in Git"}
        else:
            record(path,REPORT/relative)
    if len(pfs_only)!=4: raise ValueError("Expected raw common500 and Kodak24 frame files for two mechanisms")
    manifest={"report_root":str(REPORT),"source_root":str(EXPERIMENT),"included":included,"pfs_only":pfs_only,
              "validation_base":json.loads((temp_view/"context.json").read_text())["parent"]}
    dump_once(STUDY/"light_delivery_manifest.json",manifest)
    save_once(view/REPORT/"light_delivery_manifest.json",(STUDY/"light_delivery_manifest.json").read_bytes())
    print(len(included)+1,"ordinary light files;",len(pfs_only),"hashed frame tables retained in PFS")


if __name__=="__main__":main()
