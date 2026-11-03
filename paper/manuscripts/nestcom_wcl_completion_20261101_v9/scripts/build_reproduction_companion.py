"""Package existing reproduction records; no scientific computation is performed."""
from pathlib import Path
import csv, hashlib, io, json, zipfile

P=Path(__file__).resolve().parents[1]
R=P/'reproduction';R.mkdir(exist_ok=True)
BASE='027d5d7dcfb531f81d85b257f4f426d2c916984f'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

roots=['inputs/reproduction','inputs/policy_index_20261103',
       'inputs/timing_fairness_20261102','inputs/ada_compiled_prefill_20261103',
       'inputs/unified_calibration_20261103/compiled32',
       'inputs/unified_calibration_20261103/encoder_audit']
files=set()
for root in roots:
    files.update(p for p in (P/root).rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.npz'])
for rel in ['inputs/unified_calibration_20261103/protocol.json',
            'inputs/unified_calibration_20261103/public_directories.json',
            'inputs/unified_calibration_20261103/profile_directory.csv',
            'inputs/unified_calibration_20261103/frozen_policies.json',
            'inputs/unified_calibration_20261103/pilot_finalists.json',
            'inputs/e5/identity.json','inputs/e5/resource_ledger.json',
            'inputs/e5/source_fitting_accounting.csv','inputs/e5/model_call_accounting.csv',
            'reproduction_logs/timing_execution_history_20261103.md']:
    p=P/rel
    if p.exists():files.add(p)

# Retain all recorded environment/device objects with their exact JSON pointers.
env=[]
keys={'numeric_settings','numeric','runtime','environment','device_identity','device',
      'software_versions','versions','cpu_affinity','affinity','resources','torch_version'}
def walk(obj,ptr,source):
    if isinstance(obj,dict):
        for k,v in obj.items():
            q=ptr+'/'+k.replace('~','~0').replace('/','~1')
            if k.lower() in keys:env.append({'source':source,'json_pointer':q,'record':v})
            walk(v,q,source)
    elif isinstance(obj,list):
        for i,v in enumerate(obj):walk(v,ptr+'/'+str(i),source)
for p in sorted(files):
    if p.suffix=='.json':walk(json.loads(p.read_text(encoding='utf8')),'',p.relative_to(P).as_posix())
dump(R/'environment_device_records.json',{
    'scope':'Exact recorded fields, grouped for access; not a newly collected package freeze.',
    'base_commit':BASE,'records':env,
    'timing_protocols':{'main':'Two distinct physical CPU cores; recorded affinity [38,40]; two intra-op/two inter-op threads.',
                        'historical':'Two SMT siblings of one physical core; recorded affinity [38,39]; six intra-op/two inter-op threads.',
                        'pooling':'Never pooled.'}})

checklist='''# NeST-Com reproduction companion — 2026-11-03

This companion indexes the preserved scientific records used by the current V9 manuscript. The present revision changes signaling exposition, terminology and reading order; it performs no new inference, channel simulation, scoring, bootstrap calculation or timing measurement. Baseline repository commit: `'''+BASE+'''`.

The local/PFS delivery includes `reproduction_companion.zip`. Repository policy excludes ZIP archives from Git; the same records are tracked individually below. In a checkout, generate the ZIP with `python paper/manuscripts/nestcom_wcl_completion_20261101_v9/scripts/build_reproduction_companion.py`. It preserves these relative paths and contains the metadata, candidate-score CSVs and small reference vectors, without trained checkpoints, datasets or native image caches.

| Check | Accessible record | What to verify |
|---|---|---|
| Algorithms and field definitions | [S1 in the supplementary PDF](supplementary.pdf), [protocol](inputs/unified_calibration_20261103/protocol.json), [reproduction configuration](inputs/reproduction/reproduction_config.json) | Preconfigured MCS/directory; endpoint Profile ID; source-length field, CRC and protected body; valid NULL versus rejected frame. |
| Current and auxiliary configurations | [experiment index](inputs/policy_index_20261103/experiment_index.csv), [policy sets](inputs/policy_index_20261103/policy_sets.csv), [directory identities](inputs/policy_index_20261103/directory_versions.json) | MCS, target ceiling and exact directory version for each experiment; current results are not silently substituted for historical policies. |
| Model and code identities | [complete model manifest](inputs/reproduction/model_manifest.json) | Distinguish checkpoint-file hashes from tensor-state hashes; retain upstream commits, architecture configurations, public URLs and project storage paths. |
| Numerical environments and devices | [grouped recorded fields](reproduction/environment_device_records.json), [original model manifest](inputs/reproduction/model_manifest.json), [timing telemetry](inputs/timing_fairness_20261102/resource_telemetry_summary.json) | Exact recorded software and device settings; FP32/TF32 and CDF route; two physical cores versus historical SMT siblings. Records are not a newly collected environment freeze. |
| Decoder training recipe | [model manifest](inputs/reproduction/model_manifest.json) and repository `experiments/var-latent-enhancement-20260917/configs/experiment.json` | Frozen selected checkpoint, 20k-source adaptation and training mixture; no training in this revision. |
| Complete LDPC constructors | [24-layout CSV](reproduction/phy_layouts.csv), [original table source](reproduction/ldpc_layouts_table.tex) | Information/code lengths, base graph, lifting, filler and puncturing for the registered layouts. |
| Uniform calibration | [pilot scores](reproduction/pilot_candidate_scores.csv), [confirmation scores](reproduction/confirmation_candidate_scores.csv), [confirmation table source](reproduction/calibration_confirmation_table.tex), [frozen policies](inputs/unified_calibration_20261103/frozen_policies.json) | All candidates/SNR scores, ranks and failed/NULL outcomes; retain denominators and fixed selection rule. |
| Public profile directories | [directory JSON](inputs/unified_calibration_20261103/public_directories.json), [endpoint CSV](inputs/unified_calibration_20261103/profile_directory.csv) | Method/MCS-specific directory is known before reception; no per-image custom directory or implicit MCS switching. |
| Reference vectors | [vector manifest](inputs/reproduction/reference_vectors/reader_vectors_manifest.json), [CPU result](inputs/reproduction/reference_vectors/cpu/result.json) | Independent integer decoding, canonical finish and numerical-route limits. Small reference arrays are supplied; image caches are not. |
| Timing and paired compiled quality | [main timing](inputs/timing_fairness_20261102/timing_summary.csv), [compiled timing](inputs/ada_compiled_prefill_20261103/summary.csv), [compiled32 records](inputs/unified_calibration_20261103/compiled32/) | TX and RX measured separately; total is their sum; compiled timing remains paired with its own 32-source quality diagnostic. |
| Historical run details | [execution history](reproduction_logs/timing_execution_history_20261103.md), [historical resource identity](inputs/e5/identity.json) | Retained failures/authorization windows are provenance, not pooled scientific measurements. |
| Integrity | [SHA256 inventory](reproduction/file_inventory.csv), [machine-readable inventory](reproduction/file_inventory.json) | Hash all companion inputs and metadata before use. |

The model manifest identifies available public weights and project-trained checkpoints separately. The adapted decoder and JSCC checkpoints are retained in project storage; this package does not claim that those private PFS paths are public download links. Dataset access follows its original terms. Full local/PFS paths are provenance and may need mapping on another machine.

From the repository root, the existing deterministic reader check is:

```text
python experiments/nestcom-v9-completion-20261101/verify_cdf_ac.py --repo . --output /tmp/nest-cdf-check-fresh
```

Use a fresh output directory. Integer reference checks do not require model weights. Model/CDF regeneration additionally requires the recorded checkpoints and numerical route; set `NESTCOM_ASSET_ROOT`, or provide `NESTCOM_ASSETS_JSON` mapping checkpoint names to local paths as described in S1. This editorial revision does not execute model regeneration or claim untested heterogeneous-device interchangeability.

The companion tables are verbatim copies of the two large S1 tables moved out of the PDF. Existing main-table and result CSV bytes remain unchanged. No table values or confidence intervals are recomputed.
'''
(P/'REPRODUCTION_CHECKLIST.md').write_text(checklist,encoding='utf8')
(R/'README.md').write_text('''# Reproduction records

Start with [the manuscript companion checklist](../REPRODUCTION_CHECKLIST.md).

`phy_layouts.csv`, the pilot and confirmation score CSVs, and their two verbatim LaTeX table sources retain the numerical detail moved from S1. `environment_device_records.json` groups exact existing fields without replacing the original records. The inventory covers every included file except itself and the ZIP container.

All other metadata keep their original `inputs/` or `reproduction_logs/` paths. The local/PFS delivery includes a ZIP; Git stores the individual files and [the deterministic builder](../scripts/build_reproduction_companion.py), because archives are excluded by repository policy. Weights, datasets and native scientific image caches are not included.
''',encoding='utf8')
files.update(p for p in R.iterdir() if p.is_file() and p.name not in ['file_inventory.csv','file_inventory.json'])
files.add(P/'REPRODUCTION_CHECKLIST.md')
files.add(Path(__file__))
rows=[{'path':p.relative_to(P).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(files)]
with(R/'file_inventory.csv').open('w',encoding='utf8',newline='')as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
dump(R/'file_inventory.json',{'base_commit':BASE,'files':rows,'count':len(rows),'scientific_calls_in_this_revision':0})
files.update([R/'file_inventory.csv',R/'file_inventory.json'])
with zipfile.ZipFile(P/'reproduction_companion.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9)as z:
    for p in sorted(files):
        zi=zipfile.ZipInfo(p.relative_to(P).as_posix(),date_time=(2026,11,3,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(zi,p.read_bytes())
with zipfile.ZipFile(P/'reproduction_companion.zip')as z:
    for row in rows:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
print(json.dumps({'files':len(files),'environment_objects':len(env),'zip_bytes':(P/'reproduction_companion.zip').stat().st_size,'inventory_verified':True}))
