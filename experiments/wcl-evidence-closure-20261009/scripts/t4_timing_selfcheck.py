"""Engineering-only T4 budget/coverage tests; no fake model or PHY qualification."""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
from t4_timing_endpoint import TimingReservations,EntropyEndpoint
from t4_raw_endpoint import RawEndpoint
from t4_timing_owner import METHODS,SNRS,FIXED16,summarize,validate_native,phy

def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='wcl_t4_admission_') as directory:
        original=Path(directory)/'original.py';added=Path(directory)/'addition.py'
        original.write_text('original');added.write_text('addition')
        old={str(original):phy.sha(original)};new={str(added):phy.sha(added)}
        baseline=dict(P_native_source_bindings=old,P_native_runtime=dict(frozen_visual_identity={'model':'fixed'},numerical_runtime={'threads':6}))
        native=SimpleNamespace(loaded={'identity':{'model':'fixed'}},flags={'threads':6},driver_bindings=dict(old,**new))
        validate_native(native,baseline,old,new)
        for field,value in [('flags',{'threads':2}),('driver_bindings',old),('loaded',{'identity':{'model':'changed'}})]:
            previous=getattr(native,field);setattr(native,field,value)
            try:validate_native(native,baseline,old,new)
            except RuntimeError:pass
            else:raise AssertionError('Native identity mismatch admitted')
            setattr(native,field,previous)
        added.write_text('changed')
        try:validate_native(native,baseline,old,new)
        except RuntimeError:pass
        else:raise AssertionError('Changed registered addition admitted')
        checks.append('Native admission preserves all original hashes/models/runtime and exact pre-registered additions')
    with tempfile.TemporaryDirectory(prefix='wcl_t4_engineering_') as directory:
        p=Path(directory)/'budget.jsonl';budget=TimingReservations(p,'0'*64,2)
        budget.begin('engineering_frame');size=p.stat().st_size;called=[]
        for kind in ('header','body'):
            result=budget.call({'event_id':'engineering_frame:'+kind},{'engineering':True},
                lambda:called.append(True) or {'scientific_call':False})
            assert result=={'scientific_call':False} and p.stat().st_size==size
        assert len(called)==2;budget.finish()
        assert budget.snapshot()==dict(reserved_packet_slots=2,actual_packet_calls=2,complete_frames=1,unresolved_frames=0,cap=2)
        try:budget.begin('over_cap')
        except RuntimeError:pass
        else:raise AssertionError('Packet cap failed')
        budget.close();receipts=[json.loads(line) for line in p.read_text().splitlines()]
        assert receipts[1]['status']=='RESERVED' and receipts[-1]['status']=='COMPLETE'
        checks.append('Durable-before-call reservations; fresh callbacks; no callback-window file writes; strict cap')
        p=Path(directory)/'failure.jsonl';budget=TimingReservations(p,'1'*64,2);budget.begin('failure')
        def fail():raise ValueError('Engineering exception')
        try:budget.call({'event_id':'failure:header'},{},fail)
        except ValueError:pass
        else:raise AssertionError('Exception hidden')
        assert budget.snapshot()['unresolved_frames']==1;budget.close()
        assert json.loads(p.read_text().splitlines()[-1])['status']=='FAILED_PRESERVED'
        try:TimingReservations(p,'1'*64,2)
        except RuntimeError:pass
        else:raise AssertionError('Automatic retry allowed')
        checks.append('Failed callback remains unresolved and cannot restart into an old log')
    rows=[]
    for method in METHODS:
        for snr in SNRS:
            for i in FIXED16:
                for repeat in range(3):
                    rows.append(dict(method=method,snr_db=snr,source_id='engineering-'+str(i),phase='measured',gray=False,
                        TX_seconds=.1,RX_seconds=.2,software_e2e_excluding_channel_seconds=.31,actual_frame_energy=2048.,rho=1.,
                        header_accepted=True,body_crc_accepted=repeat!=2))
    result=summarize(rows)
    all_rows=[r for r in result if r['condition']=='all'];empty=[r for r in result if r['condition']=='gray_fallback']
    assert len(all_rows)==60 and all(r['sample_count']==48 and r['source_count']==16 and r['body_crc_rejected_samples']==16 for r in all_rows)
    assert all(r['sample_count']==0 and r['mean'] is None for r in empty)
    try:summarize(rows[:-1])
    except RuntimeError:pass
    else:raise AssertionError('Missing frame accepted')
    checks.append('Exact four-method/three-SNR/16-source/three-repeat coverage; CRC-failed KEEP frames retained; empty groups explicit')
    for method in (EntropyEndpoint.receive,RawEndpoint.receive):
        tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
        reads={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Load)}
        assert not {'last_tokens','last_tx','source_attempts','points','family'} & reads
        assert not {'tokens','pixels','tx_profile','payload','target_m'} & set(inspect.signature(method).parameters)
    checks.append('Both receive APIs and attribute reads exclude transmitter/source truth')
    print(json.dumps(dict(status='PASS_ENGINEERING_ONLY',checks=checks,actual_packet_decodes=0,neural_model_calls=0,timing_measurements=0)))
if __name__=='__main__':main()
