"""Audit closed expansion shards and publish scalar rows by endpoint only."""
from common import *
import csv,collections,statistics

def main():
    reg=read(O/'expansion_registration.json');regsha=sha(O/'expansion_registration.json')
    owner=read(L/'expansion_initial_control_v1/launch.json')['process'];assert process(owner['pid'])is None
    assert read(L/'expansion_dispatch_v1/completion.json')['all_exit0']
    out=O/'cache_expansion_delivery_v1';assert not out.exists()
    for d in reg['sources']:checked(d)
    seed=read(O/'seed_delivery_v1/closure_audit.json');checked(seed['rows']);checked(seed['seed_index'])
    with pathlib.Path(seed['rows']['path']).open()as f:allrows=list(csv.DictReader(f))
    commits=[];shards=[];seen={x['source_id']for x in allrows};exp_seen=set()
    for i in range(8):
        d=L/'expansion_dispatch_v1'/f'shard_{i:02d}';s=L/'expansion_cache_v1'/f'shard_{i:02d}'
        w=read(d/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and not w['timed_out'] and process(w['process']['pid'])is None
        assert not(s/'failure.json').exists()and not(d/'failure.json').exists();c=read(s/'completion.json');assert c['registration_sha256']==regsha
        checked(c['rows']);checked(c['ledger']);started=collections.Counter();finished=collections.Counter()
        with pathlib.Path(c['ledger']['path']).open()as f:
            for line in f:
                e=json.loads(line);assert e['status']in['STARTED','COMPLETE'];target=started if e['status']=='STARTED'else finished;target[e['kind'],e['key']]+=1
        assert started==finished and all(v==1 for v in started.values())
        counts=collections.Counter(k[0]for k in finished)
        assert counts['tx_probability_pass']==c['sources'] and all(counts[k]==8*c['sources']for k in ['independent_source_decode','predict_overwrite_advance','lpips_distance','ms_ssim'])
        expected=reg['selected_sources'][i::8];assert c['sources']==len(expected)
        for row in expected:
            p=s/f"train_i{row['source_index']:05d}"/'completion.json';r=read(p)
            for k in ['latents','wire','received']:checked(r[k])
            assert row['source_id']not in seen and row['source_id']not in exp_seen;exp_seen.add(row['source_id'])
            assert[(x['m'],x['K'])for x in r['rows']]==[tuple(e)for e in reg['endpoints']]
            assert all(x['source_id']==row['source_id'] and x['status']=='LOSSLESS_SOURCE_RX_COMPLETE'for x in r['rows'])
            allrows.extend(r['rows']);commits.append(dict(source_id=row['source_id'],source_index=row['source_index'],completion=pin(p),latents=r['latents'],wire=r['wire'],received=r['received']))
        shards.append(dict(shard=i,actual_wait=pin(d/'actual_wait.json'),completion=pin(s/'completion.json'),counts=dict(counts),sources=c['sources'],io_seconds=c['io_seconds'],source_seconds=c['source_seconds'],elapsed_seconds=c['elapsed_seconds'],peak_allocated_bytes=c['peak_allocated_bytes']))
    assert len(exp_seen)==18968 and len(allrows)==160544
    out.mkdir();index=L/'cache_expansion_index_v1.json';save(index,dict(registration=pin(O/'expansion_registration.json'),seed_index=seed['seed_index'],new_sources=commits))
    fields=list(allrows[0]);tables=[];means=[]
    for m,k in reg['endpoints']:
        rows=[r for r in allrows if int(r['m'])==m and int(r['K'])==k];rows.sort(key=lambda x:(x['split'],int(x['source_index'])))
        assert len(rows)==20068
        p=out/f'endpoint_m{m}_k{k}.csv'
        with p.open('x',newline='')as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        assert p.stat().st_size<10_000_000;tables.append(pin(p))
        for split,n in [('train',19968),('calibration',100)]:
            v=[r for r in rows if r['split']==split];assert len(v)==n
            means.append(dict(split=split,sources=n,m=m,K=k,**{metric:statistics.mean(float(r[metric])for r in v)for metric in ['terminated_bits','finish_bits','bits_per_pixel','mse','psnr_db','lpips_alex','ms_ssim']}))
    with(out/'endpoint_means.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
    save(out/'closure_audit.json',dict(status='PASS',registration=pin(O/'expansion_registration.json'),registration_compact=pin(O/'expansion_registration_compact.json'),owner_gone=True,all8children_true_exit0_and_gone=True,shards=shards,source_index=pin(index),new_train_sources=18968,new_endpoint_rows=151744,complete_8endpoint_train_sources=19968,calibration_sources=100,combined_endpoint_rows=160544,separate_benchmark_train_sources=32,separate_benchmark_endpoint_rows=64,total_unique_source_TX_passes_including_benchmark=20100,successful_science_replayed=0,scalar_tables=tables,encoder_or_VAR_updates=0,tokenizer_calls=0,PHY_calls=0))
    print(json.dumps(pin(out/'closure_audit.json')))

if __name__=='__main__':main()
