"""Post-exit truth audit only. Produces 700 exact clean-state input records."""
import argparse,csv,json,pathlib,sys,time
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from owner import BASE,O,RUN,A1,REG,desc,verify,put,success
def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def calls(root,expected):
    got={}
    for p in sorted((root/'calls').glob('*.reserved.json')):
        d=json.loads(p.read_text());kind=d['kind'];got[kind]=got.get(kind,0)+1
        assert p.with_name(p.name.replace('.reserved.json','.complete.json')).is_file()
        assert not p.with_name(p.name.replace('.reserved.json','.failed.json')).exists()
    assert got==expected,(got,expected)
    assert not list((root/'calls').glob('*.failed.json'))
def main():
    a=argparse.ArgumentParser();a.add_argument('--registration',required=True);a.add_argument('--expected-registration',required=True);args=a.parse_args()
    assert pathlib.Path(args.registration)==REG and desc(REG)['sha256']==args.expected_registration
    reg=json.loads(REG.read_text());verify(reg['sources']);sources=json.loads(pathlib.Path(reg['sources']['path']).read_text())['cal100']
    for name in ['tx','rx']:success(RUN/f'{name}.actual_wait.json')
    calls(RUN/'tx',{'model_load':2,'caption':92,'tokenizer':92,'tx_probability':92})
    calls(RUN/'rx',{'model_load':1,'rx_prefix':676,'rx_render':676})
    tx=json.loads((RUN/'tx/completion.json').read_text());rx=json.loads((RUN/'rx/completion.json').read_text())
    assert not tx['blocked_reads'] and not rx['blocked_reads'] and len(tx['outputs'])==0 and len(rx['outputs'])==676
    previous=json.loads((BASE/'a1_closed_files.json').read_text());oldpins={d['path']:d for d in previous['files']}
    oldrows={d['key']:d for d in json.loads((A1/'rx/completion.json').read_text())['outputs']}
    newrows={d['key']:d for d in rx['outputs']};assert len(newrows)==676 and len(oldrows)==24
    lengths=[];states=[];proofs=[]
    for s in sources:
        i=s['rank'];sid=f'{i:02d}';troot=A1/'tx' if i<8 else RUN/'tx'
        metadata=troot/(sid+'_source.json')
        if i<8:verify(oldpins[str(metadata)])
        data=json.loads(metadata.read_text());assert data['source_id']==s['source_id'] and data['source_index']==s['source_index']
        assert len(data['lengths'])==7 and len(data['cdf_hashes'])==7
        for e in range(1,8):
            key=f'{sid}_e{e}';inherited=i<8 and e in [1,4,7];r=(oldrows if inherited else newrows)[key]
            assert r['endpoint']==e and r['caption_bytes_hex']==data['caption_bytes_hex'] and r['cdf_hashes']==data['cdf_hashes'][:e]
            image=pathlib.Path(r['path']);assert image==(A1 if inherited else RUN)/'rx'/(key+'.npz')
            if inherited:
                verify(oldpins[str(image)]);image_pin=oldpins[str(image)]
            else:
                with np.load(troot/(sid+'_truth.npz'),allow_pickle=False) as truth,np.load(RUN/'rx'/(key+'_decoded.npz'),allow_pickle=False) as received:
                    assert set(received.files)=={f's{j}' for j in range(1,e+1)}
                    assert all(np.array_equal(truth[f's{j}'],received[f's{j}']) for j in range(1,e+1)),key
                with np.load(image,allow_pickle=False) as z:
                    pix=z['pixels'];floats=z['float_output'];assert pix.shape==(3,256,256) and pix.dtype==np.uint8 and np.isfinite(floats).all()
                    import hashlib
                    assert hashlib.sha256(pix.tobytes()).hexdigest()==r['pixel_sha256']
                image_pin=desc(image)
            wire=troot/(key+'.arc');wire_pin=oldpins[str(wire)] if i<8 else desc(wire)
            if i<8:verify(wire_pin)
            item=data['lengths'][e-1];assert item['endpoint']==e and item['payload_bits']==wire_pin['bytes']*8 and item['payload_sha256']==wire_pin['sha256']
            lengths.append({'rank':i,'source_id':s['source_id'],'source_index':s['source_index'],**item,'outer_length_bits':13,'outer_crc_bits':16})
            states.append({'key':key,'rank':i,'source_id':s['source_id'],'source_index':s['source_index'],'endpoint':e,
              'source':s['archive'],'source_pixels_sha256':data['pixels_sha256'],'image':image_pin,'pixel_sha256':r['pixel_sha256'],
              'wire':wire_pin,'generation_seed':0,'origin':'A1_EXACT_REUSE' if inherited else 'A2_NEW_INDEPENDENT_RX'})
            proofs.append({'key':key,'source_id':s['source_id'],'endpoint':e,'origin':states[-1]['origin'],'caption_exact':True,'CDF_exact':True,'tokens_exact':True,'truth_used_after_rx_exit_only':True})
    assert len(states)==len(lengths)==700 and len({r['key'] for r in states})==700
    write_csv(O/'source_lengths.csv',lengths);write_csv(O/'independent_prefix_audit.csv',proofs)
    put(O/'clean_states.json',{'schema':'ARPC_A2_CLEAN_STATES_V1','registration_sha256':args.expected_registration,'states':states})
    closed=[]
    for root in [RUN/'tx',RUN/'rx',RUN/'wire_inputs']:
        for p in sorted(root.rglob('*')):
            if p.is_file() and 'runtime_cache' not in p.parts:closed.append(desc(p))
    put(O/'closed_files.json',{'files':closed})
    report={'status':'CLEAN_STATES_AUDITED_NO_METRICS_OR_WIRELESS','registration_sha256':args.expected_registration,'ended_unix':time.time(),
      'total_sources':100,'new_caption_tokenizer_TX_each':92,'reused_source_passes':8,'new_rx_prefix_and_render_each':676,'reused_clean_states':24,
      'total_clean_states':700,'wireless_frames':0,'metric_calls':0,'training_updates':0,'unresolved':0,
      'outputs':[desc(O/n) for n in ['source_lengths.csv','independent_prefix_audit.csv','clean_states.json','closed_files.json']]}
    put(O/'completion.json',report);print(json.dumps(report))
if __name__=='__main__':main()
