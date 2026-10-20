"""Closed-score statistics using the exact existing common500 index bank."""
from common import *
import itertools,collections,shutil,sqlite3,traceback,sys

def main(expected):
 reg=read(O/'registration.json');assert sha(O/'registration.json')==expected
 for d in reg['code']:checked(d)
 for control in ['score_control','initial_control']:
  wait=read(T/control/'actual_wait.json')
  assert type(wait['returncode'])is int and wait['returncode']==0 and wait['actual_wait']is True
  assert gone(wait['process'])and gone(wait['parent_process'])
  for k in ['stdout','stderr']:checked(wait[k])
  assert not(T/control/'failure.json').exists()
 assert not(T/'score/failure.json').exists()
 done=read(T/'score/completion.json')
 assert done['registration_sha256']==expected and done['logical_pairs']==22500 and done['unresolved']==0
 assert done['unique_features']==reg['limits']['image_embedding']
 assert done['unique_pairs']==reg['limits']['cosine_pair']
 inv=read(checked(reg['inventory']));sources=read(checked(inv['sources_file']));frames=read(checked(inv['frames_file']))
 grid(frames,sources)
 for entry in read(checked(done['features'])):checked(entry['feature'])
 with checked(done['frame_scores']).open()as f:rows=list(csv.DictReader(f))
 indexed={key(v):v for v in rows}
 assert len(rows)==len(indexed)==22500 and set(indexed)=={key(v)for v in frames}
 for v in frames:
  row=indexed[key(v)]
  assert row['source_id']==v['source_id']and row['reference_sha256']==v['reference_sha256']and row['pixel_sha256']==v['pixel_sha256']
  assert row['receiver_status']==v['receiver_status']and(row['fixed_gray']=='True')==v['fixed_gray']
 bank=np.load(checked(inv['bootstrap']['indices']),allow_pickle=False)
 assert bank.shape==(10000,500)and ash(bank)==inv['bootstrap']['array_sha256']
 out=O/'delivery';out.mkdir()
 source_rows=[];arrays={};summary=[];paired=[]
 for method in METHODS:
  for snr in SNRS:
   a=[]
   for i in range(500):
    group=[float(indexed[method,i,snr,n]['clip_image_cosine'])for n in SEEDS[method]]
    value=float(np.mean(group,dtype=np.float64));a.append(value)
    source_rows.append(dict(method=method,source_index=i,source_id=sources[i]['source_id'],
        snr_db=snr,metric='clip_image_cosine',source_mean=value,noise_count=3))
   a=np.asarray(a,dtype=np.float64);assert np.isfinite(a).all()
   arrays[method,snr]=a
   summary.append(dict(method=method,snr_db=snr,metric='clip_image_cosine',direction='higher',source_count=500,**interval(a,bank)))
 for first,second in itertools.combinations(METHODS,2):
  for snr in SNRS:
   d=interval(arrays[first,snr]-arrays[second,snr],bank)
   paired.append(dict(method=first,reference=second,snr_db=snr,metric='clip_image_cosine',
       source_count=500,noise_per_source=3,**d,interval_crosses_or_touches_zero=d['ci_low']<=0<=d['ci_high']))
 assert len(source_rows)==7500 and len(summary)==15 and len(paired)==30
 shutil.copyfile(checked(done['frame_scores']),out/'frame_scores.csv')
 csvsave(out/'source_means.csv',source_rows);csvsave(out/'summary.csv',summary);csvsave(out/'paired_differences.csv',paired)
 nest=[v for v in paired if v['method']==METHODS[0]]
 csvsave(out/'nest_minus_baselines.csv',nest)
 counts=[]
 for method in METHODS:
  for snr in SNRS:
   rr=[v for v in rows if v['method']==method and int(v['snr_db'])==snr]
   counts.append(dict(method=method,snr_db=snr,logical_pairs=len(rr),sources=len({v['source_id']for v in rr}),
       noise_count=3,fixed_gray=sum(v['fixed_gray']=='True'for v in rr),
       unique_pixel_pairs=len({(v['reference_sha256'],v['pixel_sha256'])for v in rr})))
 csvsave(out/'denominators.csv',counts)
 oldpath=R/'results/arpc_common500_extension_20261019_v1/delivery_v1/common500_summary.csv'
 pairpath=R/'results/arpc_common500_extension_20261019_v1/delivery_v1/nest_minus_arpc_paired.csv'
 with oldpath.open()as f:old=list(csv.DictReader(f))
 with pairpath.open()as f:oldpairs=list(csv.DictReader(f))
 assert len(old)==105 and len(oldpairs)==21
 # Preserve original seven-metric numeric strings verbatim; append CLIP only.
 csvsave(out/'eight_metric_summary.csv',old+summary)
 additional=[]
 for p in nest:
  if p['reference']!=METHODS[4]:continue
  additional.append(dict(comparison='NeST-Com_entropy_minus_ARPC',snr_db=p['snr_db'],metric='clip_image_cosine',
      direction='higher',source_count=500,mean=p['mean'],ci_low=p['ci_low'],ci_high=p['ci_high'],
      nest_favorable_sign='positive',interval_crosses_or_touches_zero=p['interval_crosses_or_touches_zero']))
 csvsave(out/'eight_metric_nest_minus_arpc.csv',oldpairs+additional)
 # Review exact preservation without re-evaluating any original statistic.
 with(out/'eight_metric_summary.csv').open()as f:combined=list(csv.DictReader(f))
 with(out/'eight_metric_nest_minus_arpc.csv').open()as f:combinedpairs=list(csv.DictReader(f))
 assert combined[:105]==old and combinedpairs[:21]==oldpairs
 receipt=out/'receipts';receipt.mkdir()
 for root,names in {
 'inventory_control':['launch.json','actual_wait.json','stdout.txt','stderr.txt'],
 'qualification':['launch.json','actual_wait.json','stdout.txt','stderr.txt'],
 'score_control':['launch.json','actual_wait.json','stdout.txt','stderr.txt'],
 'initial_control':['launch.json','actual_wait.json','stdout.txt','stderr.txt'],
 'score':['completion.json','metric_identity.json','resource_history.json']}.items():
  target=receipt/root;target.mkdir()
  for n in names:shutil.copyfile(T/root/n,target/n)
 compute=dict(logical_pairs=22500,source_count=500,SNR_count=3,noise_count=3,
   unique_reconstruction_pixels=inv['unique_reconstruction_pixels'],source_embeddings=500,
   actual_image_embeddings=done['unique_features'],actual_cosine_pairs=done['unique_pairs'],
   exact_logical_pair_reuses=22500-done['unique_pairs'],previous_CLIP_scores_reused=0,
   new_model_loads=1,new_caption=0,new_channel=0,new_reconstruction=0,new_other_neural_metrics=0,
   new_bootstrap_indices=0,existing_bootstrap=inv['bootstrap'],source_mean_rows=7500,
   marginal_rows=15,direct_paired_rows=30,unchanged_old_summary_cells=105,unchanged_old_ARPC_paired_cells=21)
 save(out/'actual_compute_and_reuse.json',compute)
 table=['| Method | 4 dB | 10 dB | 19 dB |','|---|---:|---:|---:|']
 for m in METHODS:
  cells=[next(v for v in summary if v['method']==m and v['snr_db']==s)for s in SNRS]
  table.append('| '+NAMES[m]+' | '+' | '.join(f"{v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]"for v in cells)+' |')
 comparisons=['| Reference | SNR (dB) | NeST − reference | Direct 95% CI |','|---|---:|---:|---:|']
 for v in nest:
  comparisons.append(f"| {NAMES[v['reference']]} | {v['snr_db']} | {v['mean']:.5f} | [{v['ci_low']:.5f}, {v['ci_high']:.5f}] |")
 text=f"""# CLIP-I: current five-method N1024 common500 supplement

All22,500 logical pairs were scored; every method/SNR retains500 sources and
1500 frames. The byte/pixel inventory found no missing native array. NeST19dB
uses the T1-C m10-corrected current PARTIAL result. Final P100k/Swin200k,
adaptive BPG and all4500 ARPC receiver records are preserved.

CLIP-I is **CLIP image-image cosine similarity**, higher is better. It uses
OpenAI CLIP ViT-L/14,224 input, the original floating tensor preprocessing and
FP32 inference. Both inputs are images; values are raw cosines. This is distinct
from caption-oriented [CLIPScore](https://aclanthology.org/2021.emnlp-main.595/).
No caption, softmax or percentage conversion is involved.

## Five-method means and pointwise95% source-bootstrap intervals

{chr(10).join(table)}

## Direct source-paired NeST-minus-baseline differences

{chr(10).join(comparisons)}

The original three noise realizations are averaged within each source before
equal-source aggregation. The existing seed2026100701,10000×500 index bank is
reused without drawing new indices. Differences are formed at source level
before resampling. These are pointwise intervals, not simultaneous or
multiple-comparison-adjusted claims. Historical baseline noise namespaces differ;
pairing is by source, not by an identical cross-method noise sample. All thirty
signed method-pair contrasts are reported in paired_differences.csv.

## Actual compute and reuse

{done['unique_features']} image embeddings =500 originals +
{inv['unique_reconstruction_pixels']} distinct native reconstructed images.
{done['unique_pairs']} distinct source/reconstruction cosine scores serve22,500
logical positions; {22500-done['unique_pairs']} positions reuse an exact pair.
There were zero new training, channel, decoder, reconstruction, caption or
other-neural-metric calls. All original105 seven-metric summary cells and21
NeST-minus-ARPC paired cells are copied without numeric alteration into the
eight-metric combined tables. The existing main figures remain unchanged.

The actual inputs in this complete grid contain zero fixed-gray frames;
the inventory and statistics retain the original receive statuses and never
filter by success. Actual length fallback endpoints remain in frame_scores.csv.
The independent four selected/final two development examples are excluded.
N2048 has not been supplemented in this first-stage delivery.

Frozen weights and seven official source files were restored from the existing
static backup with every original SHA verified; the inference environment was
not upgraded. New feature caches, ledgers and native arrays remain outside Git.
Registration, input/file hashes, model/runtime identity and actual exit receipts
provide the lineage. Prior ARPC engineering telemetry limitations remain as
documented in its original delivery; this evaluation does not relabel them.
"""
 (out/'REPORT.md').write_text(text,encoding='utf8')
 tex=["% CLIP-I raw image-image cosine, not percent. Requires booktabs.",
      r"\begin{tabular}{lccc}",r"\toprule",r"Method & 4 dB & 10 dB & 19 dB \\",r"\midrule"]
 for m in METHODS:
  vv=[next(v for v in summary if v['method']==m and v['snr_db']==s)for s in SNRS]
  tex.append(NAMES[m]+' & '+' & '.join(f"{v['mean']:.4f}"for v in vv)+r' \\')
  tex.append(' & '+' & '.join(f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]"for v in vv)+r' \\')
 tex.extend([r'\bottomrule',r'\end{tabular}'])
 (out/'clip_i_table.tex').write_text('\n'.join(tex)+'\n')
 files=[desc(p)for p in sorted(out.rglob('*'))if p.is_file()]
 save(out/'manifest.json',dict(files=files,registration_sha256=expected,original_summary=desc(oldpath),original_paired=desc(pairpath)))
 save(out/'completion.json',dict(status='COMPLETE_CLIP_I_FIVE_METHOD_COMMON500',registration_sha256=expected,
     manifest=desc(out/'manifest.json'),compute=compute,score_completion=desc(T/'score/completion.json'),
     old_scores_unchanged=True,N2048_run=False,development_pooled=False))
 print(json.dumps(compute),flush=True)
if __name__=='__main__':main(sys.argv[1])
