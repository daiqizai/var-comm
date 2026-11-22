"""Record the actual Git versions and independently compare archived result CSVs to PFS."""
import hashlib
import json
from pathlib import Path

from study import REPO, STUDY, dump_once, git, regular_path

ROOT="results/swinjscc_mismatch_zpac_20261120/"
HISTORY={
  "V1":("f5bb7c4dd06f76457912cf102701b09b4c62bb57",
        [ROOT+x for x in ("eval_common500_main","eval_common500_ablation","eval_common500_fading","eval_kodak24_all")]),
  "V2":("7a2d1dec5cc19b958c9ccc68d4894cb8d82595b9",[ROOT+"v2_eval"]),
  "V3":("f7d764847ba3abd6e6fb4102f04b89e9b1412611",[ROOT+"v3_blind_eval"]),
  "V4":("65a5a8e14b92eb7fad92eb6e6444418e52d9a073",[ROOT+"v4_universal_eval"]),
  "3A,3B":("165b6f200e38f37766b499111f3f9c419aa0b086",["results/dir3_mimo_holorate_20261121/eval"]),
  "4A,4B":("ec608c7d89d78704b4efa7d32004fafec85f6766",["results/dir4_rf1bit_noma_20261122/eval"]),
}


def main():
    result={}
    for label,(rev,dirs) in HISTORY.items():
        files={}
        for p in git("ls-tree","-r","--name-only",rev,*dirs).decode().splitlines():
            if not p.endswith(".csv"):
                continue
            path=REPO/p
            regular_path(path)
            raw=git("show",f"{rev}:{p}")
            assert raw==path.read_bytes(),p
            files[p]=hashlib.sha256(raw).hexdigest()
        assert files
        result[label]={"git_commit":rev,"csv_count":len(files),"sha256":files}
    dump_once(STUDY/"historical_inputs.json",result)
    print({k:v["csv_count"] for k,v in result.items()})


if __name__=="__main__": main()
