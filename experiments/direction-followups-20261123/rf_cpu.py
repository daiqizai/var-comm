"""Port only the vendor attention-mask device placement for CPU RF inference."""
import functools
from pathlib import Path
import sys
from common import STUDY, TEMP, digest, include_sources, path_check, record, save, single_replace

include_sources()
import rf as experiment
from dir2_zpac.models import DEFAULT_VENDOR_ROOT, load_swinjscc_backbone

VENDOR = TEMP / "cpu_vendor"


def prepare():
    manifest = {}
    for part in ("net","loss"):
        for src in sorted((DEFAULT_VENDOR_ROOT/part).rglob("*.py")):
            rel = src.relative_to(DEFAULT_VENDOR_ROOT)
            code = path_check(src).read_bytes()
            if rel.as_posix()=="net/encoder.py":
                code = single_replace(code.decode(),"self.attn_mask = attn_mask.cuda()",
                                      "self.attn_mask = attn_mask.to(self.norm1.weight.device)").encode()
            compile(code,str(src),"exec")
            save(VENDOR/rel,code)
            manifest[str(rel)] = {"original":digest(src),"cpu":digest(VENDOR/rel)}
    assert [x for x,v in manifest.items() if v["original"]!=v["cpu"]] == ["net/encoder.py"]
    record(STUDY/"rf/cpu_placement_revision.json",{
        "files":manifest,"original_failure":"Unconditional .cuda() in SwinTransformerBlock.update_mask before any image or channel was processed.",
        "revision":"Put the same integer-region attention mask (values 0 and -100) on the block's parameter device. Frozen parameters, quantizer, PHY, sample identities and scientific protocol unchanged.",
    })


if __name__ == "__main__":
    if sys.argv[1]=="prepare":
        prepare()
    else:
        from common import read
        for rel,ids in read(STUDY/"rf/cpu_placement_revision.json")["files"].items():
            assert digest(VENDOR/rel) == ids["cpu"]
        experiment.load_swinjscc_backbone = functools.partial(load_swinjscc_backbone,vendor_root=VENDOR)
        experiment.execute(sys.argv[1],"cpu")
