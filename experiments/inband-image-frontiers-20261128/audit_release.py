"""Verify counts, identities, actual exits and plain files in the scientific delivery."""
import csv
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parent
RUNTIME = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/inband_image_frontiers_20261128_v1")


def digest(path):
    with path.open("rb") as f:
        sha=hashlib.sha256()
        for block in iter(lambda:f.read(1<<20),b""):sha.update(block)
        return sha.hexdigest()


def read(file):
    return json.loads(file.read_text())


def main():
    workers={}
    ids={}
    results={}
    expected={"activity/mask_decisions.csv":8000,"separation/common500_groups.csv":4500,
              "separation/val256_groups.csv":576,"separation/solo.csv":750,
              "frontend/common500_events.csv":20000,"drift/continuous_events.csv":12000,"drift/qam64_events.csv":12000}
    for role in ("activity","separation","frontend","drift"):
        r=ROOT/role
        job=RUNTIME/("job_"+role)
        reg=read(job/"registration.json");proc=read(job/"process.json");wait=read(job/"wait.json")
        assert wait["actual_wait"]==0 and wait["wall_seconds"]<=reg["limit"]
        statfile=Path("/proc")/str(proc["child"])/"stat"
        assert not statfile.exists() or statfile.read_text().split()[21]!=proc["start_ticks"]
        assert all(digest(ROOT/n)==v for n,v in reg["sources"].items())
        for split,n in (("common500",500),("val256",256)):
            path=r/(split+".json")
            if split=="val256" and role=="activity":continue
            src=read(path)
            assert len(src["ids"])==len(set(src["ids"]))==len(set(src["pixels"]))==n
            ids.setdefault(split,src)
            assert ids[split]==src
        identity=read(r/"identity.json")
        workers[role]={"registration":reg,"process":proc,"actual_exit":wait,"gpu":identity["gpu"],
                   "torch":identity["pytorch"],"cuda":identity["cuda"],"cores":identity["cores"],
                   "tf32_matmul":identity["tf32_matmul"],"tf32_cudnn":identity["tf32_cudnn"],"counts":read(r/"complete.json")}
    assert set(ids["val256"]["pixels"]).isdisjoint(ids["common500"]["pixels"])
    for rel,n in expected.items():
        with (ROOT/rel).open() as f: assert sum(1 for _ in csv.DictReader(f))==n, rel
    for folder in (*workers,"analysis"):
        for p in sorted((ROOT/folder).iterdir()):
            st=p.lstat()
            assert stat.S_ISREG(st.st_mode) and st.st_nlink==1 and not p.is_symlink()
            results[str(p.relative_to(ROOT))]={"bytes":st.st_size,"sha256":digest(p)}
    ledger={"protocol_sha256":digest(ROOT/"protocol.json"),"workers":workers,
            "elapsed_device_hours_reserved_sum":sum(w["actual_exit"]["wall_seconds"] for w in workers.values())/3600,
            "device_hours_definition":"Sum of per-worker elapsed wall time including initialization; not GPU kernel active time.",
            "per_role_gpu_limit_seconds":5400,"new_training_steps":0,"common500":500,"disjoint_validation":256,
            "event_counts":expected,"files":results}
    with (ROOT/"ledger.json").open("x") as f:
        json.dump(ledger,f,indent=2,ensure_ascii=False)


if __name__=="__main__":
    main()

