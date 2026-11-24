"""Export the pinned receiver from Git objects and add three optional observation points."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REV = "faaa2799f67593f1fe7846a4b421701a19d81f4d"
PACKAGES = (
    "swinjscc-mismatch-zpac-20261120", "dir3-mimo-holorate-20261121",
    "dir4-rf1bit-noma-20261122", "dir5-otfs-genvar-20261122",
    "dir6-matched-breakthroughs-20261123",
)
MODEM = f"experiments/{PACKAGES[-1]}/src/dir6_mb/overloaded_noma_4x2.py"


def optional_hooks(source):
    changes = [
        (
            "                p_clean_u = (2.0 * z_clean_u.square()",
            "                replica = z_clean_u\n"
            "                if getattr(self, 'study_trace', None) is not None:\n"
            "                    replica = self.study_trace.replica(sic_round, u, ch_u, cr_u, z_clean_u)\n"
            "                p_clean_u = (2.0 * replica.square()",
        ),
        ("self.project_data_perp(z_clean_u / p_clean_u[:, None, None])",
         "self.project_data_perp(replica / p_clean_u[:, None, None])"),
        ("                omega_list.append(omega_um)",
         "                if getattr(self, 'study_trace', None) is not None:\n"
         "                    omega_um = self.study_trace.gate(sic_round, u, omega_um)\n"
         "                omega_list.append(omega_um)"),
        ("            y_sic_r = y_perp0_r[:, None] - (tot_r - contrib_r)",
         "            if getattr(self, 'study_trace', None) is not None:\n"
         "                self.study_trace.capture((tot_r - contrib_r)[:, 3], (tot_i - contrib_i)[:, 3])\n"
         "            y_sic_r = y_perp0_r[:, None] - (tot_r - contrib_r)"),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError(f"Unexpected upstream instrumentation site: {before}")
        source = source.replace(before, after)
    return source


def prepare(repo, dest):
    def read(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    prefix = [f"experiments/{s}/src" for s in PACKAGES]
    names = read("ls-tree", "-r", "--name-only", REV, "--", *prefix).decode().splitlines()
    names = [n for n in names if n.endswith(".py")]
    assert MODEM in names
    assets = {}
    for name in names:
        raw = read("show", f"{REV}:{name}")
        out = optional_hooks(raw.decode()).encode() if name == MODEM else raw
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(out)
        assets[name] = {"upstream": hashlib.sha256(raw).hexdigest(), "deployed": hashlib.sha256(out).hexdigest()}
    for split in ("common500", "kodak24"):
        path = f"results/dir6_matched_breakthroughs_20261123/eval_dir6c_{split}.json"
        (dest / f"anchor_{split}.json").write_bytes(read("show", f"{REV}:{path}"))
    (dest / "snapshot.json").write_text(json.dumps({"commit": REV, "files": assets}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    prepare(args.repo, args.dest)
