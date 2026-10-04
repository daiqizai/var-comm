"""Fixed16 exploratory HiFi/Swin scope; preserve all original source indices."""
from __future__ import annotations
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent/'runtime'))
sys.path.insert(0,str(HERE.parent))
from external_eval_common import *
import external_eval_common as paired
from fixed80k_adapter import paths as fixed_paths,require,CHECKPOINT_SHA,STEP
SOURCE_INDICES=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
SNRS=(7,13);SEEDS=(2001,);BUDGETS=(1024,2048);SOURCES=16;PHYSICAL_FRAMES=64;ROWS=128
VERSION='USER-FIXED80K-HIFI-FIXED16-20261004-R1'
FILES=('hifi16_common.py','hifi16_eval.py','hifi16_score.py','test_hifi16.py')

def locations(root):
    p=fixed_paths(root);child=p['revision']/'hifi_fixed16_release'
    return dict(root=p['root'],child=child,config=child/'config.json',output=child/'evaluation',
        result=p['root']/'results/external_comparison_20261004/fixed80k_revision/hifi_fixed16_release/evaluation',
        parent_config=p['config'],parent_output=p['output'],runtime=p['runtime'])

def frame_specs(source_index):
    require(type(source_index)is int and source_index in SOURCE_INDICES,'Only the registered original fixed16 indices are allowed')
    return [dict(source_index=source_index,N=n,snr_db=s,noise_seed=2001) for n in BUDGETS for s in SNRS]

def expected_ids(source_index):
    return [row_id(spec,m) for spec in frame_specs(source_index) for m in METHODS]

def code_bindings():
    import external_eval
    return {**external_eval.code_bindings(),**{str(HERE/name):sha(HERE/name) for name in FILES}}

def validate_config(config):
    p=locations(config['root']);require(config==read(p['config']) and config['version']==VERSION,'Fixed16 config differs')
    require(config.get('source_indices')==list(SOURCE_INDICES) and config.get('methods')==list(METHODS)
        and config.get('source_count')==16 and config.get('rows')==128 and config.get('physical_frames')==64
        and config.get('total_N')==[1024,2048] and config.get('snrs')==[7,13] and config.get('noise_seeds')==[2001]
        and config.get('full_comparison_complete') is False and config.get('full_queue_resume_allowed') is False,
        'Fixed16 scope cannot become the full evaluation')
    require(Path(config['output'])==p['output'] and Path(config['result'])==p['result'],'Fixed16 isolated output differs')
    verify(config['fixed16_source_bindings'])
    parent=read(p['parent_config']);normalized=dict(config)
    for key in ('version','output','result','source_count','snrs','noise_seeds'):normalized[key]=parent[key]
    for key in ('source_indices','methods','rows','physical_frames','full_comparison_complete','full_queue_resume_allowed',
        'fixed16_source_bindings','paired_cache_snapshot','user_request_path'):normalized.pop(key)
    require(normalized==parent,'Original paid PHY/checkpoint/full posterior protocol changed')
    import external_eval
    external_eval.validate_config(parent)
    return p['root']

def selected_sources(baseline):
    identities={}
    for row in baseline:
        sid=str(row['source_id']);index=int(row['source_index']);prediction=int(row['resnet50_source_prediction']);truth=int(row['true_class_index'])
        require(sid not in identities and index in SOURCE_INDICES and 0<=prediction<1000 and 0<=truth<1000,'Fixed16 original identity differs')
        require(float(row['resnet50_source_top1_label'])==float(prediction==truth) and row['preprocessing_id'],'Original classifier baseline differs')
        identities[sid]=dict(index=index,preprocessing_id=row['preprocessing_id'],reference_sha256=row['reference_sha256'],prediction=prediction,truth=truth)
    require(len(identities)==16 and {v['index'] for v in identities.values()}==set(SOURCE_INDICES),'Fixed16 original population incomplete')
    return sorted(identities,key=lambda sid:identities[sid]['index']),identities

def validate_source(value, binding, source_index):
    if (value.get("binding") != binding or value.get("source_index") != source_index
            or value.get("payload_sha256") != identity({k: v for k, v in value.items() if k != "payload_sha256"})):
        raise RuntimeError("Source checkpoint identity/checksum differs")
    rows = value["rows"]
    if [r["replay_row_id"] for r in rows] != expected_ids(source_index):
        raise RuntimeError("Source checkpoint lacks exact 8-row coverage")
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
        if slots != proof["image_slots"] or len(slots) != 8 or any(type(s) is not int or not 0 <= s < len(images) for s in slots):
            raise RuntimeError("Source reconstruction slots differ")
        for row, slot in zip(rows, slots):
            if row["image_sha256"] != rgb_sha(images[slot]) or row["reference_sha256"] != rgb_sha(target):
                raise RuntimeError("Source cache differs from scientific pixels")
    return value

