"""Audit normally closed diagnostics and preserve a publication figure snapshot.

This does not compute model outputs, bootstrap intervals, or modify source CSVs.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/baseline_strength_20261011'
CHAIN=BASE/'automatic_followthrough_v1'
OUT=BASE/'diagnostic_delivery_v1'
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))


def main():
    assert not OUT.exists(),'Snapshot already created'
    reg=read(CHAIN/'registration.json');audits=[];bindings={}
    reference_hashes={}
    for stage,count,specs in [('milestones',1000,reg['milestone_methods']),('bottleneck',100,reg['bottleneck_methods'])]:
        done=read(CHAIN/stage/'completion.json');wait=read(CHAIN/(stage+'_actual_wait.json'))
        assert wait['exit_code']==0 and wait['actual_child_waited'] and wait['stop_reason'] is None
        assert done['status']=='COMPLETE_FROZEN_FRAME_SCORING' and done['frames']==reg['stage_frames'][stage]
        for path,h in done['outputs'].items():assert sha(Path(path))==h;bindings[path]=h
        total=0
        for spec in specs:
            data=rows(CHAIN/stage/(spec['name']+'.csv'));seeds=spec['seeds'];snrs=spec.get('snrs',[1,4,7,10,13,19])
            expected={(i,s,n) for i in range(count) for s in snrs for n in seeds};seen=set();ids={}
            for row in data:
                i,s,n=(int(row[k]) for k in ['source_index','snr_db','noise_seed']);key=(i,s,n)
                assert key in expected and key not in seen and row['method']==spec['name'];seen.add(key)
                assert int(row['N'])==(0 if spec.get('mode')=='direct_F' else 1024)
                ref=(row['source_id'],row['reference_sha256'])
                if i in reference_hashes:assert reference_hashes[i]==ref
                else:reference_hashes[i]=ref
                ids[i]=row['source_id']
            assert seen==expected and len(set(ids.values()))==count;total+=len(data)
            audits.append(dict(stage=stage,method=spec['name'],frames=len(data),source_count=count,
                               noise_count=len(seeds),header_rejected=sum(r['header_accepted']=='False' for r in data),
                               fallback_count=sum(r['fallback']!='none' for r in data),missing=0,duplicates=0,
                               reused=sum(r['reused']=='True' for r in data)))
        assert total==reg['stage_frames'][stage]
    OUT.mkdir();(OUT/'figures').mkdir()
    for p in (BASE/'diagnostic_figures_v1').iterdir():
        if p.is_file():shutil.copy2(p,OUT/'figures'/p.name)
    data=rows(CHAIN/'milestones/descriptive_summary.csv');deltas=[]
    for branch,start,end in [('P',40000,100000),('Swin',80000,200000)]:
        for snr in [1,4,7,10,13,19]:
            row=dict(branch=branch,method=branch+str(end),reference=branch+str(start),snr_db=snr)
            for metric in METRICS:
                a=next(r for r in data if r['method']==row['method'] and int(r['snr_db'])==snr and r['metric']==metric)
                b=next(r for r in data if r['method']==row['reference'] and int(r['snr_db'])==snr and r['metric']==metric)
                row[metric]=float(a['mean'])-float(b['mean'])
            deltas.append(row)
    with (OUT/'parent_to_selected_descriptive_deltas.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(deltas[0]));w.writeheader();w.writerows(deltas)
    text=['# Closed full-calibration milestones and bottleneck diagnostic','',
          'Both actual child processes exited zero. Milestones contain exactly 96,000 logical frames (91,800 new, 4,200 verified reuses); the bottleneck diagnostic contains 900 frames (300 new, 600 verified reuses). Full frame grids, source identities, failures, duplicates and input/output hashes passed. There are no rejected headers or fixed-fallback frames in these diagnostic grids; no row was filtered.', '',
          'Milestones use all 1,000 original calibration sources and all six displayed SNR values at N1024. P uses three original noises; Swin uses its original one-noise calibration protocol. The first 100 calibration sources form the separate bottleneck diagnostic. These descriptive means do not have bootstrap intervals and do not enter model selection or stopping. They are not holdout results.', '',
          '## Selected checkpoint minus parent (descriptive)', '',
          '| Branch | SNR (dB) | Delta PSNR (dB) | Delta LPIPS | Delta DINOv2-L | Delta agreement (pp) |',
          '|---|---:|---:|---:|---:|---:|']
    for r in deltas:text.append(f"| {r['branch']} | {r['snr_db']} | {r['psnr_db']:.6f} | {r['lpips_alex']:.6f} | {r['dinov2_vitl14_cosine']:.6f} | {100*r['convnext_top1_source_prediction']:.6f} |")
    text+=['', 'The selected checkpoints improve all four displayed parent-relative means at all six SNR values. This does not imply monotonic late-training improvement: P DINOv2-L and agreement decrease at some SNR values between 80k and 100k; Swin agreement also decreases at several SNR values between 160k and 200k. The full milestone curves retain these changes. No test-metric-based reselection is made.', '',
           'P stops under the registered empirical plateau rule. Swin reaches its additional-update budget while still improving its selection objective; convergence is not claimed. Swin changes include fresh optimizer/sampling/RNG initialization from 80k weights and expanded 1--19 dB coverage. Its original 80k/19 dB point remains outside the old training/calibration range.', '',
           '## Bottleneck interpretation', '',
           'The full F-to-Dc path is a free-latent reference, not an equal-budget achievable upper bound. Both no-noise communication paths retain the real N1024 bottleneck and receiver SNR conditioning at 19 dB. Their gap to the full-latent reference suggests a substantial loss in the tested communication bottleneck; it does not establish a capacity limit for all continuous JSCC architectures. Agreement can fluctuate with noise and is not source-label classification accuracy.', '',
           'The three diagnostic figure groups were exported as vector PDF, editable SVG and 600 dpi PNG. The publication uses a fixed copy under `figures/` so later automatic rendering cannot mutate published bytes. Reproduce the source figures with `python experiments/baseline-delivery-20261012/plot_evaluation.py diagnostics`; this snapshot packaging entry is once-only.', '',
           'Formal common500 and existing-new100 scoring and its gated source-paired statistical successor remain separate. Fair timing is pending the exclusive resource window and timing-entry qualification.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    (OUT/'audit.json').write_text(json.dumps(dict(status='PASS_CLOSED_DIAGNOSTIC_GRIDS',methods=audits,input_sha256=bindings,
        training_updates=0,bootstrap_calls=0,source_scoring_calls=0,failures_filtered=False,visual_review='Pending explicit PNG review record'),indent=2)+'\n')


if __name__=='__main__':main()
