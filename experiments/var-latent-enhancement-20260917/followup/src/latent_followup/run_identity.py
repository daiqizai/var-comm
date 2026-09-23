"""Small follow-up binding helpers using the existing B snapshot conventions."""
import json
from pathlib import Path
import torch
from latent_enhancement.runtime import digest, verify_snapshot, write_json, CONFIG, EXPERIMENT
from latent_enhancement_b.common import bind_files, decoder_gate_path, OUT_B

ARM_KEYS={'original_residual_control':'control','predicted_innovation_candidate':'candidate'}

def checked_checkpoint(record, root):
    p=Path(record.get('checkpoint',record.get('path','')))
    p=p if p.is_absolute() else Path(root)/p
    expected=record.get('checkpoint_sha256',record.get('sha256'))
    if not expected or digest(p)!=expected: raise RuntimeError('checkpoint SHA mismatch')
    return p

def selection_record(name, selection, checkpoint, parent_step):
    return {**selection,'arm_key':ARM_KEYS[name],'checkpoint':str(checkpoint),
            'checkpoint_sha256':digest(checkpoint),'parent_step':int(parent_step),
            'incremental_step':int(selection['step']),'total_step':int(parent_step)+int(selection['step'])}

def register_run(output, config, parents):
    output=Path(output); path=output/'registration.json'
    files=[CONFIG,Path(config),decoder_gate_path(),OUT_B/'rx_cache/completion.json',*map(Path,parents)]
    for folder in ['src','phase_b/src','evaluation/src','followup/src','mechanisms/src']:
        files.extend(sorted((EXPERIMENT/folder).rglob('*.py')))
    identity={'source_bindings':bind_files(files)}
    if path.exists():
        saved=json.loads(path.read_text());verify_snapshot(saved['source_bindings'])
        if saved!=identity:raise RuntimeError('run configuration/source/parent scope changed')
    else:
        if (output/'latest.json').exists():raise RuntimeError('unbound historical resume refused; use a new evaluation run')
        write_json(path,identity)
    return digest(path)

def load_resume(record, root, registration_sha):
    p=checked_checkpoint(record,root)
    payload=torch.load(p,map_location='cpu',weights_only=True)
    if payload.get('registration_sha256')!=registration_sha:raise RuntimeError('checkpoint registration mismatch')
    return payload
