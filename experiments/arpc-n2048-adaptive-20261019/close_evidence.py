"""Exact copies of closed receipts and a pending-scope delivery index."""
from shared import *
import fcntl,sqlite3

def main():
 lock=(L/'execution.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(O/'B0_B1_delivery/completion.json')['status']=='PACKAGE_REQUIRED_PENDING_RESULT_DELIVERY'
 review=O/'figures_calibration_v2/actual_visual_review.json';assert read(review)['status']=='PASS'
 copies=[];processes=[];expected_fail={'qualification_rx'}
 def copy(src,dest):
  assert src.is_file()and not src.is_symlink()and src.stat().st_size<7000000
  before=pin(src);raw=src.read_bytes();verify(before);dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(dest)==before['sha256'];copies.append(dict(source=before,copy=pin(dest)))
 def control(c,dest):
  if (c/'launch.json').exists():
   w=read(c/'actual_wait.json');q=read(c/'launch.json')
   assert type(w['returncode'])is int and w['returncode']==(1 if c.name in expected_fail or (c.name=='rx'and c.parent.name=='arpc_n2048_adaptive_b1_transport_control_v1')else 0)
   assert not any(w.get(k)for k in ('timed_out','terminated','killed','metadata_error'))
   assert gone(q['process'])and gone(q['parent_process'])
   processes.append(dict(control=str(c),launch=pin(c/'launch.json'),wait=pin(c/'actual_wait.json'),child_gone=True,parent_gone=True))
  for p in sorted(c.iterdir()):
   assert not p.is_symlink()
   if p.is_dir():control(p,dest/p.name)
   elif p.is_file():copy(p,dest/p.name)
 for c in sorted((L/'controls').iterdir()):
  if c.name=='close_evidence':continue
  control(c,O/'execution_receipts'/c.name)
 for name in ('arpc_n2048_adaptive_b0_control_v1','arpc_n2048_adaptive_engineering_control_v1','arpc_n2048_adaptive_b1_transport_control_v1','arpc_n2048_adaptive_saved_closure_control_v1'):
  control(BASE/'baseline_strength_20261011'/name,O/'outer_receipts'/name)
 for relative in ('qualification/tx/completion.json','qualification/rx/failure.json','qualification_rx_closure_v1/completion.json','qualification_constructor_diagnostic_v1/diagnosis.json','cal100_verified_cache_v1/completion.json','engineering/completion.json','cal100_scores_v1/worker/completion.json','cal100_scores_v1/worker/metric_identity.json','cal100_scores_v1/worker/scores.json'):
  src=L/relative
  if src.exists():copy(src,O/'runtime_receipts'/relative)
 ledgers=[]
 for relative in ('qualification/tx/calls.sqlite','qualification/rx/calls.sqlite','qualification/native/rx/calls.sqlite','cal100_clean_v1/rx/calls.sqlite','cal100_scores_v1/worker/calls.sqlite'):
  p=L/relative;assert p.is_file()
  db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);rows=db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status').fetchall();assert all(x[1]=='COMPLETE'for x in rows);db.close()
  ledgers.append(dict(source=pin(p),rows=rows))
 put(O/'closed_receipts_manifest.json',dict(status='PASS_EXACT_BYTE_COPIES',copies=copies,process_closure=processes,ledger_summaries=ledgers,new_scientific_calls=0,active_arrays_copied=False))
 text='''# N2048 low-SNR adaptation: B0/B1 delivered, B2/B3 pending

The package-required uncertainty stop applies. This is a completed calibration
and qualification delivery, not a completed adaptive wireless comparison.

Cal100 now contains3200 clean source endpoints (2500 NeST-Com and700 ARPC).
All actual terminated source bytes were present; no TX or tokenizer rerun.
1407 existing image/score states were verified;1793 missing NeST states and
1793 DINO values were completed once.24 qualification objects used48 subframes
and96 decodes with zero repeated science. Original RX final exit1 and its
metadata-only closure are preserved in execution_receipts and outer_receipts.

At10dB five NeST-Com MCS policies remain plausible under the coarse historical
single-packet proxy: QPSK2/3,3/4,5/6 and16-QAM1/2,2/3. A maximum of two does
not cover them. B2 and B3 have not started. No new wireless quality, paired
bootstrap, seven-metric test result or final winner is claimed.

- B0_B1_delivery/REPORT_zh.md: full Chinese report, pending policies and costs.
- B0_B1_delivery/additional_costs.csv: three extra-policy ceilings relative to
  the eight-proposal plan (9000 logical objects/18000subframes/36000decodes).
- A tighter coverage plan retains only ARPC's one plausible policy at each
  SNR: nine total policy/SNR cells,27000logical/54000subframes/108000decodes.
  This still exceeds the original global limits by3000/6000/12000 and requires
  five NeST-Com candidates at10dB instead of the authorized two. NOT EXECUTED.
- calibration_diagnostics/:3200 source-level clean scores,32 endpoint means,
  all384 policy utilization rows and actual fallback endpoint distributions.
- cal100_prescreen_v1/: all384 strategies at4/10dB with frozen uncertainty rule.
- figures_calibration_v2/: actual opened PDF/editableSVG/600dpiPNG diagnostic,
  grayscale proof, English caption and root visual review. v1 is preserved;
  only the grayscale legend color was corrected in v2.
- layouts.csv and arpc_source_cost_breakdown_detailed.csv: both fragments pay
  their68-symbol header and13-bit length+16-bitCRC+96-bit reassembly record.
- qualification/: actual independent source-RX outputs and opened-image review.
- closed_receipts_manifest.json: exact copied evidence and gone-process audit.

The historical N1024 and fixed-full N2048 results are unchanged.19dB may only
be shown as its separate completed fixed-full reference. No new training,
power extension, UEP, feedback, KEEP, extra methods or heavy backups. Further
science awaits a scope/budget decision; this task does not silently drop any
plausible policy. Publication receipts are delivered separately after push.
'''
 (O/'README.md').open('x',encoding='utf8').write(text)
 put(O/'final_delivery_manifest.json',dict(status='PACKAGE_REQUIRED_PENDING_RESULT_DELIVERY_COMPLETE',science=pin(O/'B0_B1_delivery/completion.json'),diagnostic_tables=pin(O/'calibration_diagnostics/manifest.json'),final_figure=pin(O/'figures_calibration_v2/manifest.json'),actual_visual_review=pin(review),closed_receipts=pin(O/'closed_receipts_manifest.json'),index=pin(O/'README.md'),B2=False,B3=False,new_budget_authorized=False,publication_pending=True,backup='DEFERRED'))
 print(json.dumps(dict(status='PENDING_DELIVERY_CLOSED',copies=len(copies),processes=len(processes))))

if __name__=='__main__':main()
