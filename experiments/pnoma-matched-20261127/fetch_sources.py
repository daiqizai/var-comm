"""Fetch only the public neural network modules at pinned author revisions."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

PNOMA = "f2b7777130cf7242ecbb773c168b6080a89e630e"
COMPRESSAI = "79e51871e972c517bc528b689ccf15137fcc7785"
FILES = {
    "ipc-lab/deepjscc-pnoma": (PNOMA, ["LICENSE", "README.md", "configs/model/base_deepjscc.yaml",
        "src/models/nets/deepjsccq2.py", "src/models/nets/utils.py", "src/models/components/afmodule.py",
        "src/models/noma/deepjscc_noma_orth.py"]),
    "InterDigitalInc/CompressAI": (COMPRESSAI, ["LICENSE","compressai/layers/layers.py",
        "compressai/layers/gdn.py","compressai/ops/parametrizers.py","compressai/ops/bound_ops.py"])
}

def fetch(repo, rev, path, git_sha, dest):
    url = f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"
    data = urlopen(Request(url, headers={"User-Agent":"var-comm-research"}),timeout=40).read()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != git_sha:
        raise ValueError(f"Git blob identity mismatch for {url}")
    local = dest / (path if path.endswith(".py") else f"origins/{repo.split('/')[1]}/{path}")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        assert local.read_bytes() == data
    else:
        local.write_bytes(data)
    return str(local.relative_to(dest)).replace("\\","/"),dict(url=url,git_blob=blob,sha256=hashlib.sha256(data).hexdigest())

def main(dest):
    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = []
        for repo,(rev, paths) in FILES.items():
            tree=json.load(urlopen(Request(f"https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1",
                headers={"User-Agent":"var-comm-research"}),timeout=30))
            entries={e["path"]:e for e in tree["tree"]}
            jobs.extend(pool.submit(fetch,repo,rev,n,entries[n]["sha"],dest) for n in paths)
        pins=dict(j.result() for j in jobs)
    shims = {"compressai/layers/__init__.py":"from .layers import AttentionBlock, ResidualBlock, ResidualBlockUpsample, ResidualBlockWithStride\n"}
    for p in ("src", "src/models", "src/models/nets", "src/models/components", "compressai", "compressai/ops"):
        shims[p+"/__init__.py"] = ""
    for name,content in shims.items():
        p=dest/name
        p.parent.mkdir(exist_ok=True,parents=True)
        p.write_bytes(content.encode())
    (dest/"vendor_identity.json").write_text(json.dumps(dict(upstream=pins,namespace_shims=shims),indent=2,sort_keys=True)+"\n")
    print(len(pins),"verified blobs")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dest",type=Path)
    main(parser.parse_args().dest)
