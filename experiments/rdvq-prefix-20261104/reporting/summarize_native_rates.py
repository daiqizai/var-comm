"""Summarize existing byte lengths and rate audits; no codec or model calls."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from packet import SourcePacket
BUDGETS=[384,512,640,768,1024,1280,1536,2048,2560,3072,4096]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_write(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def describe(values):
    x=np.asarray(values,dtype=float)
    return dict(n=int(x.size),minimum=float(x.min()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),
                mean=float(x.mean()),p95=float(np.quantile(x,.95)),maximum=float(x.max()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    L=Path(a.root);O=Path(a.output);assert read(L/'p1_source_gate.json')['passed'];O.mkdir(exist_ok=False)
    m=L/'p1_registration_v1/evaluation_manifest.json';sources=read(m)['sources'];groups={s:[] for s in ['development','calibration','test']};raw={};overview=[]
    decreases=ties=neighbors=0
    for source in sources:
        rows=read(source['lengths_path']);assert len(rows)==28 and [v['slices'] for v in rows]==list(range(1,29))
        for v in rows:
            if 'entropy_stream_bits' not in v:
                # The eight reused P0 rows have a normalized length-only schema.
                # Read their existing public packet headers; no entropy/model RX.
                assert source['reuse_p0']
                packet=SourcePacket.from_bytes((Path(source['public_directory'])/f"s{v['slices']:02d}.bin").read_bytes())
                accounting=packet.accounting();assert accounting['packet_bits']==v['packet_bits'];v.update(accounting)
            assert v['packet_bits']==v['entropy_stream_bits']+72
            assert v['entropy_stream_bits']==v['rans_state_bits']+v['entropy_renormalization_bits']
        seq=np.array([72]+[v['packet_bits'] for v in rows]);diff=np.diff(seq);decreases+=int((diff<0).sum());ties+=int((diff==0).sum());neighbors+=len(diff)
        sid=source['source_id'];groups[source['split']].append(sid);raw[sid]=dict(bits=seq,overhead=np.array([72]+[72+v['rans_state_bits'] for v in rows]))
    for split,ids in groups.items():
        for endpoint in [1,4,12,14,20,28]:
            overview.append(dict(split=split,slices=endpoint,**describe([raw[s]['bits'][endpoint] for s in ids])))
    csv_write(O/'packet_length_distribution.csv',overview)
    overhead=[]
    for b in BUDGETS:
        ids=groups['test'];selection=[max(i for i,x in enumerate(raw[s]['bits']) if x<=b) for s in ids]
        bit=np.array([raw[s]['bits'][i] for s,i in zip(ids,selection)]);ov=np.array([raw[s]['overhead'][i] for s,i in zip(ids,selection)])
        overhead.append(dict(budget_bits=b,sources=len(ids),mean_packet_bits=float(bit.mean()),mean_selected_slices=float(np.mean(selection)),
              null_count=sum(i==0 for i in selection),full_count=sum(i==28 for i in selection),hard_budget_violations=int((bit>b).sum()),
              mean_wrapper_fraction_of_packet=float(np.mean(72/bit)),mean_wrapper_plus_rANS_states_fraction_of_packet=float(np.mean(ov/bit)),
              mean_fraction_of_source_budget_used=float(np.mean(bit/b))))
    csv_write(O/'selected_packet_overhead.csv',overhead)
    r0=L/'p1_r0_rate_proxy_v1/complete.json';initial=read(r0);records=initial['records']
    assert len(records)==128 and [v['source_id'] for v in records]==groups['development']
    actual=np.array([v['actual_packet_bits'] for v in records]);estimated=np.array([v['estimated_packet_bits'] for v in records])
    correction=np.array(initial['median_packet_minus_proxy_bits']);assert actual.shape==estimated.shape==(128,29)
    rate=[];selection_rows=[]
    for t in range(1,29):
        rate.append(dict(slices=t,median_applied_correction=float(correction[t]),**describe(actual[:,t]-estimated[:,t])))
    csv_write(O/'initial_rate_proxy_error.csv',rate)
    for name,pred in [('estimated_only',estimated),('in_sample_corrected',estimated+correction[None])]:
        for b in BUDGETS:
            true=np.where(actual<=b,np.arange(29)[None],0).max(1);chosen=np.where(pred<=b,np.arange(29)[None],0).max(1)
            used=actual[np.arange(128),chosen]
            selection_rows.append(dict(proxy=name,budget_bits=b,sources=128,endpoint_matches=int((chosen==true).sum()),
                proxy_selected_overbudget=int((used>b).sum()),underselected=int((chosen<true).sum()),overselected=int((chosen>true).sum()),
                mean_absolute_slice_error=float(np.abs(chosen-true).mean()),scope='development audit, correction evaluated in sample; final fitting always checks actual bytes'))
    csv_write(O/'initial_proxy_endpoint_diagnostic.csv',selection_rows)
    result=dict(source_manifest_sha256=sha(m),initial_rate_audit_sha256=sha(r0),sources=len(sources),independent_nonnull_packets=len(sources)*28,
                compared_adjacent_lengths=neighbors,decreasing_adjacent_lengths=decreases,equal_adjacent_lengths=ties,
                statement='Observed adjacency counts are empirical only. The native rANS adapter still enumerates actual independently terminated candidates without assuming monotonicity.',
                metric_calls=0,model_calls=0,new_statistics=0,PHY_calls=0)
    (O/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
