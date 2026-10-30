"""Matched real LDPC/QAM timing only; repeated benchmark calls do not add quality trials."""
from opt_common import *
from frame_worker import Packets
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];cases=cr(r['cases']);rx=cr(r['rx_cases']);expected_cases={v['case_id']:v for v in rx};out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']))
 total=len(cases)*4*5;ledger=Ledger(out/'calls.sqlite',dict(header=total,body=total));start=time.monotonic();save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'));import c_predclass_frame as cp
  sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'));import n512_phy as n5
  sys.path.insert(0,str(R/'experiments/wcl-adatoken-20261027'));from public_counter_rx import receive
  phy=cp.isolated_engine();ns=phy.pack_body.__globals__;ns['FAMILIES']=(*ns['FAMILIES'][:-2],cp.FAMILY,cp.NULL);rt=phy.create_runtime(R,str(checked(r['qualification'])));cat=cr(r['catalogue']);rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']};rows=[]
  for grid in ['quarter','eighth','token','E']:
   for case in cases:
    j=case['physical_cases'][grid];bits=np.asarray(j['payload'],np.uint8);assert asha(bits)==j['payload_sha256'];noise=phy.standard_noise(j['source_id'],j['noise_seed']);scaled=noise*10**(-j['snr_db']/20)
    for phase,count in [('warmup',2),('measured',3)]:
     for repeat in range(count):
      if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('PHY timing cap')
      event=f"{grid}:{case['case_id']}:{phase}:{repeat}";beg=time.perf_counter();wave,tx=rt.transmit(j['profile_id'],bits,j['counter']);txsec=time.perf_counter()-beg;observed=wave+scaled;beg=time.perf_counter()
      if j['snr_db']==7:v=receive(phy,rt,observed,7,j['public_base'],Packets(ledger),event,phase='BENCHMARK_ONLY')
      else:v=rt.receive(observed,j['snr_db'],j['counter'],rt.profiles,Packets(ledger),event,phase='BENCHMARK_ONLY')
      rxsec=time.perf_counter()-beg;projection=n5.received_projection(v);assert projection==expected_cases[case['case_id']]['projections'][grid]
      rows.append(dict(input_grid=grid,case_id=case['case_id'],source_id=case['source_id'],N=case['N'],snr_db=case['snr_db'],phase=phase,repeat=repeat,tx_PHY_seconds=txsec,rx_PHY_seconds=rxsec,status=v['status']))
   print('TIMING_PHY',grid,len(rows),flush=True)
  assert not rt.torch.cuda.is_initialized();save(out/'rows.json',rows);save(out/'completion.json',dict(status='COMPLETE',rows=len(rows),actual_PHY_calls_purpose='timing repetitions only, no new quality samples',warmups=2,measured=3,ledger=ledger.summary(),seconds=time.monotonic()-start,CUDA_initialized=False))
 except BaseException as e:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
