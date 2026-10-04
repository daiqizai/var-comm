"""Swin-only early release: original physical/cache contracts, one real method."""
from __future__ import annotations
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent/'runtime'))
sys.path.insert(0,str(HERE.parent))
from external_eval_common import *
import external_eval_common as paired
from fixed80k_adapter import paths as fixed_paths, require, CHECKPOINT_SHA, STEP
METHODS=(paired.METHODS[0],)
ROWS=1800
VERSION='USER-FIXED80K-SWIN-EARLY-20261004-R1'
FILES=('swin_early_common.py','swin_early.py','swin_early_score.py','test_swin_early.py')

def locations(root):
    p=fixed_paths(root);child=p['revision']/'swin_early_release'
    return dict(root=p['root'],child=child,output=child/'evaluation',config=child/'config.json',
        result=p['root']/'results/external_comparison_20261004/fixed80k_revision/swin_early_release/evaluation',
        parent_config=p['config'],parent_output=p['output'],runtime=p['runtime'])

def expected_ids(source_index):
    return [row_id(spec,METHODS[0]) for spec in frame_specs(source_index)]

def code_bindings():
    import external_eval
    return {**external_eval.code_bindings(),**{str(HERE/name):sha(HERE/name) for name in FILES}}

def validate_config(config):
    p=locations(config['root']);saved=read(p['config'])
    require(config==saved and config.get('version')==VERSION,'Swin early config differs')
    require(config.get('methods')==list(METHODS) and config.get('rows')==1800
        and config.get('full_comparison_complete') is False,'Only real 1800-row Swin evaluation is registered')
    require(Path(config['output'])==p['output'] and Path(config['result'])==p['result'], 'Swin early paths differ')
    parent=read(p['parent_config'])
    original=dict(config)
    for key in ('version','output','result'):original[key]=parent[key]
    for key in ('methods','rows','full_comparison_complete','early_source_bindings','paired_cache_snapshot','user_request_path'):
        original.pop(key)
    require(original==parent,'Original paid PHY/model/population config changed')
    verify(config['early_source_bindings'])
    import external_eval
    external_eval.validate_config(parent)
    return p['root']

def validate_frame_receipt(value, binding, spec, source):
    require(value.get('hifi_evaluated') is False, 'Swin-only release cannot contain a HiFi result')
    if value.get("binding") != binding or value.get("frame") != spec:
        raise RuntimeError("Frame resume identity differs")
    if value.get("payload_sha256") != identity({k: v for k, v in value.items() if k != "payload_sha256"}):
        raise RuntimeError("Frame receipt checksum differs")
    if value.get("source_id") != source["image_id"] or value.get("reference_sha256") != rgb_sha(source["rgb"]):
        raise RuntimeError("Frame source pixels/identity differ")
    rows = value["rows"]
    if [r["replay_row_id"] for r in rows] != [row_id(spec, m) for m in METHODS]:
        raise RuntimeError("Frame does not contain the exact Swin-only method")
    if sha(value["archive"]) != value["archive_sha256"]:
        raise RuntimeError("Committed physical frame archive changed")
    with np.load(value["archive"], allow_pickle=False) as archive:
        if set(archive.files) != {"images", "observed", "transmitted"}:
            raise RuntimeError("Physical frame cache schema differs")
        images, observed, signal = archive["images"], archive["observed"], archive["transmitted"]
        if images.shape != (1, 3, 256, 256):
            raise RuntimeError("Physical frame cached shape differs")
        if observed.shape != (spec["N"], 2) or signal.shape != (spec["N"], 2):
            raise RuntimeError("Physical waveform count differs")
        if array_sha(observed) != value["observed_sha256"] or array_sha(signal) != value["transmitted_sha256"]:
            raise RuntimeError("Paired physical waveform changed")
        if not np.isfinite(observed).all() or abs(float(np.square(signal).sum()) - 2 * spec["N"]) > .02:
            raise RuntimeError("Physical waveform energy/finite check differs")
        for i, row in enumerate(rows):
            require(row.get('selected_step') == STEP and row.get('selected_checkpoint_sha256') == CHECKPOINT_SHA
                and row.get('NFE') == 0 and row.get('complete_posterior_schedule') is False,
                'Swin-only frame must use exact80k with no posterior sampling')
            if row.get("method") != METHODS[i] or any(row.get(k) != v for k, v in spec.items()):
                raise RuntimeError("Scientific frame fields do not match the actual pair")
            if rgb_sha(images[i]) != row["image_sha256"] or row["observed_sha256"] != value["observed_sha256"]:
                raise RuntimeError("Scored pixels or shared observation identity differ")
            if (type(row.get("header_accepted")) is not bool
                    or row["header_accepted"] != rows[0]["header_accepted"]
                    or row.get("audit_used_to_control_receiver") is not False):
                raise RuntimeError("Receiver decisions or offline-only audit boundary differ")
            if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                   for k in ("TX_seconds", "RX_seconds")):
                raise RuntimeError("Actual TX/RX timing must be finite and nonnegative")
            if not row["header_accepted"] and not np.all(images[i] == np.float32(.5)):
                raise RuntimeError("Failed header reconstruction differs from registered gray")
    return value


def validate_source(value, binding, source_index):
    if (value.get("binding") != binding or value.get("source_index") != source_index
            or value.get("payload_sha256") != identity({k: v for k, v in value.items() if k != "payload_sha256"})):
        raise RuntimeError("Source checkpoint identity/checksum differs")
    rows = value["rows"]
    if [r["replay_row_id"] for r in rows] != expected_ids(source_index):
        raise RuntimeError("Source checkpoint lacks exact 18-row coverage")
    verify(value["frame_bindings"])
    proof = value["float_reconstructions"]
    if sha(proof["path"]) != proof["sha256"]:
        raise RuntimeError("Source floating reconstruction archive changed")
    with np.load(proof["path"], allow_pickle=False) as data:
        if set(data.files) != {"images", "source_rgb", "row_ids", "image_slots"}:
            raise RuntimeError("Source floating cache schema differs")
        if data["row_ids"].tolist() != expected_ids(source_index):
            raise RuntimeError("Source cache row IDs differ")
        images, target, slots = data["images"], pixels(data["source_rgb"]), data["image_slots"].tolist()
        if slots != proof["image_slots"] or len(slots) != 18 or any(type(s) is not int or not 0 <= s < len(images) for s in slots):
            raise RuntimeError("Source reconstruction slots differ")
        for row, slot in zip(rows, slots):
            if row["image_sha256"] != rgb_sha(images[slot]) or row["reference_sha256"] != rgb_sha(target):
                raise RuntimeError("Source cache differs from scientific pixels")
    return value

