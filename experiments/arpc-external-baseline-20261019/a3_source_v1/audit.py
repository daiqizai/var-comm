"""Close only the fixed evaluation source traversals; no wireless claim."""
import argparse,csv,json,pathlib
import control as c
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True);a=p.parse_args();assert c.desc(c.REG)['sha256']==a.expected_registration
    c.success(c.RUN/'tx.actual_wait.json');reg=json.loads(c.REG.read_text());sources=json.loads(pathlib.Path(reg['sources']['path']).read_text())['new_sources'];assert len(sources)==100
    calls=c.RUN/'tx/calls';counts={}
    for f in calls.glob('*.reserved.json'):
        kind=json.loads(f.read_text())['kind'];counts[kind]=counts.get(kind,0)+1;assert f.with_name(f.name.replace('.reserved.','.complete.')).exists()
    assert not list(calls.glob('*.failed.json')) and counts=={'model_load':2,'caption':100,'tokenizer':100,'tx_probability':100}
    done=json.loads((c.RUN/'tx/completion.json').read_text());assert not done['outputs'] and not done['blocked_reads'] and done['counts']==counts
    catalogue=[];rows=[]
    for s in sources:
        i=s['source_index'];d=json.loads((c.RUN/'tx'/f'{i:02d}_source.json').read_text());assert d['source_id']==s['source_id'] and d['source_index']==i and len(d['lengths'])==7;streams=[]
        for e in range(1,8):
            pin=c.desc(c.RUN/'tx'/f'{i:02d}_e{e}.arc');item=d['lengths'][e-1];assert item['endpoint']==e and item['payload_sha256']==pin['sha256'] and item['payload_bits']==pin['bytes']*8
            streams.append({'endpoint':e,'wire':pin,'payload_bits':pin['bytes']*8,'origin':'A3_NEW_SINGLE_TX_PASS'})
            rows.append({'source_id':s['source_id'],'source_index':i,'endpoint':e,'payload_bits':pin['bytes']*8,'wire_sha256':pin['sha256']})
        catalogue.append({'source_id':s['source_id'],'source_index':i,'source':s['archive'],'streams':streams})
    c.put(c.OUT/'stream_catalogue.json',{'schema':'ARPC_EVAL100_SOURCE_STREAMS_V1','registration_sha256':a.expected_registration,'sources':catalogue})
    with (c.OUT/'source_lengths.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    c.put(c.OUT/'completion.json',{'status':'EVAL100_SOURCE_READY_NO_WIRELESS_YET','new_source_passes':100,'streams':700,'counts':counts,'unresolved':0,'new_images':0,'new_packets':0,'new_scores':0,
      'max_reserved_bytes':done['max_reserved_bytes'],'outputs':[c.desc(c.OUT/n) for n in ['stream_catalogue.json','source_lengths.csv']]})
if __name__=='__main__':main()
