"""Closed TX source inventory; source truth never enters actual RX."""
import argparse,csv,json,pathlib
import control as c
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True);a=p.parse_args();assert c.desc(c.REG)['sha256']==a.expected_registration
    c.success(c.RUN/'tx.actual_wait.json');reg=json.loads(c.REG.read_text());source=json.loads(pathlib.Path(reg['sources']['path']).read_text())
    calls=c.RUN/'tx/calls';counts={}
    for f in calls.glob('*.reserved.json'):
        kind=json.loads(f.read_text())['kind'];counts[kind]=counts.get(kind,0)+1
        assert f.with_name(f.name.replace('.reserved.','.complete.')).exists()
    assert not list(calls.glob('*.failed.json')) and counts=={'model_load':2,'caption':900,'tokenizer':900,'tx_probability':900}
    done=json.loads((c.RUN/'tx/completion.json').read_text());assert not done['outputs'] and not done['blocked_reads'] and done['counts']==counts
    previous=json.loads((c.O/'a2_clean_v1/clean_states.json').read_text())['states'];old={(s['source_id'],s['endpoint']):s for s in previous}
    oldids={s['source_id'] for s in source['reused_sources']};newids={s['source_id'] for s in source['new_sources']};assert len(oldids)==100 and len(newids)==900 and not oldids&newids
    catalogue=[];rows=[]
    for s in source['full_sources']:
        i=s['source_index'];sid=s['source_id'];streams=[]
        if sid in newids:
            meta=c.RUN/'tx'/f'{i:02d}_source.json';d=json.loads(meta.read_text());assert d['source_id']==sid and d['source_index']==i and len(d['lengths'])==7
        for e in range(1,8):
            if sid in oldids:
                pin=old[sid,e]['wire'];c.verify(pin);origin='A1_A2A_EXACT_CLOSED_REUSE'
            else:
                pin=c.desc(c.RUN/'tx'/f'{i:02d}_e{e}.arc');item=d['lengths'][e-1]
                assert item['endpoint']==e and item['payload_sha256']==pin['sha256'] and item['payload_bits']==pin['bytes']*8
                origin='A2B_NEW_SINGLE_TX_PASS'
            streams.append({'endpoint':e,'wire':pin,'payload_bits':pin['bytes']*8,'origin':origin})
            rows.append({'source_id':sid,'source_index':i,'endpoint':e,'payload_bits':pin['bytes']*8,'wire_sha256':pin['sha256'],'origin':origin})
        catalogue.append({'source_id':sid,'source_index':i,'source':s['archive'],'streams':streams})
    assert len(catalogue)==1000 and len(rows)==7000
    c.put(c.OUT/'stream_catalogue.json',{'schema':'ARPC_CAL1000_SOURCE_STREAMS_V1','registration_sha256':a.expected_registration,'sources':catalogue})
    with (c.OUT/'source_lengths.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    c.put(c.OUT/'completion.json',{'status':'SOURCE1000_READY_NO_WIRELESS_YET','new_source_passes':900,'reused_source_passes':100,'full_sources':1000,'streams':7000,
      'counts':counts,'unresolved':0,'new_images':0,'new_packets':0,'new_scores':0,'max_reserved_bytes':done['max_reserved_bytes'],
      'outputs':[c.desc(c.OUT/n) for n in ['stream_catalogue.json','source_lengths.csv']],'source_truth_exposed_to_RX':False})
if __name__=='__main__':main()
