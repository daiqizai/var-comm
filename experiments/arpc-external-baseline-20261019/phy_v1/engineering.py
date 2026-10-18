"""Separate TX and RX for the fixed24 engineering frames; no neural calls."""
import argparse,hashlib,json,os,pathlib,sys,time,traceback
import numpy as np
import link
E=link.R/'experiments/arpc-external-baseline-20261019';sys.path.insert(0,str(E));from a1_owner import L,O,desc,put,verify
RUN=L/'a2_phy_engineering_v1';OUT=O/'a2_phy_engineering_v1';REG=OUT/'registration.json'
class Ledger:
    def __init__(self,root,caps):self.root=root;root.mkdir();self.caps=caps;self.counts={}
    def call(self,kind,key,fn):
        n=self.counts.get(kind,0)+1;assert n<=self.caps[kind];self.counts[kind]=n;p=self.root/f'{kind}_{n:03d}';t=time.time()
        put(p.with_suffix('.reserved.json'),{'kind':kind,'key':key,'started_unix':t})
        try:
            value=fn();put(p.with_suffix('.complete.json'),{'seconds':time.time()-t});return value
        except BaseException:put(p.with_suffix('.failed.json'),{'error':traceback.format_exc(),'seconds':time.time()-t});raise
def tx(reg):
    target=RUN/'tx';target.mkdir();incoming=RUN/'received_inputs';incoming.mkdir()
    cache=target/'runtime_cache';cache.mkdir();(cache/'tmp').mkdir()
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
    import tempfile;tempfile.tempdir=str(cache/'tmp')
    rt=link.Runtime();assert rt.backend.identity==reg['backend_identity'];ledger=Ledger(target/'calls',{'transmit':24});jobs=[];outputs=[]
    put(incoming/'catalogue.json',rt.catalogue)
    for item in reg['frames']:
        key=item['key'];p=rt.profile(item['endpoint'],item['q'],item['nominal_rate'])
        if item['wire'] is None:payload=b''
        else:verify(item['wire']);payload=pathlib.Path(item['wire']['path']).read_bytes()
        frame,receipt=ledger.call('transmit',key,lambda:rt.transmit(payload,p['profile_id'],item['counter']))
        noise=link.ep.standard_noise(item['source_id'],item['noise_seed'])*10**(-item['snr_db']/20)
        y=frame+noise
        with (incoming/(key+'.npy')).open('xb') as f:np.save(f,y,allow_pickle=False)
        jobs.append({'key':key,'snr_db':item['snr_db'],'counter':item['counter'],'observation':desc(incoming/(key+'.npy'))})
        outputs.append({'key':key,'transmission':receipt,'payload_sha256':hashlib.sha256(payload).hexdigest(),'noise_sha256':link.ep.array_sha(noise),'observed_sha256':link.ep.array_sha(y)})
    assert len(outputs)==24;put(incoming/'jobs.json',jobs);put(target/'completion.json',{'status':'COMPLETE','outputs':outputs,'counts':ledger.counts,'neural_calls':0})
def rx():
    target=RUN/'rx';target.mkdir();cache=target/'runtime_cache';cache.mkdir();(cache/'tmp').mkdir()
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
    import tempfile;tempfile.tempdir=str(cache/'tmp')
    from a1_native import install_rx_boundary
    rtroot=link.R.parent.parent/'var_comm_runtime_20261010'
    allowed=[E,link.OLD,link.R/'experiments/wcl-evidence-closure-20261009/scripts',link.R/'experiments/prior-aware-grouped-mcs-v1',
      link.R/'experiments/main-raw64-20261007',link.R/'src',link.R/'outputs/build',
      link.R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',rtroot/'envs/frozen_ldpc_py311',RUN/'received_inputs',target]
    blocked=install_rx_boundary(allowed)
    rt=link.Runtime();assert rt.catalogue==json.loads((RUN/'received_inputs/catalogue.json').read_text())
    sys.path.insert(0,str(E/'upstream'));import wire
    ledger=Ledger(target/'calls',{'header':24,'body':24});jobs=json.loads((RUN/'received_inputs/jobs.json').read_text());assert len(jobs)==24
    output=[]
    for job in jobs:
        verify(job['observation']);y=np.load(job['observation']['path'],allow_pickle=False)
        result=rt.receive(y,job['snr_db'],job['counter'],ledger,job['key']);body=result['body']
        if body is not None and body['payload'] is not None:
            payload=body.pop('payload');body['payload_hex']=payload.hex()
            if result['status']=='PAYLOAD_PARSED':
                try:
                    caption,blocks,modes=wire.unpack(payload,result['profile']['endpoint'])
                    result['source_syntax']='PASS';result['actual_caption_utf8_hex']=caption.encode().hex()
                except wire.SourceParseError as error:
                    result['source_syntax']='SOURCE_PARSE_FAIL';result['source_error']=str(error);result['status']='SOURCE_PARSE_FAIL'
        result.update(key=job['key'],observation_sha256=link.ep.array_sha(y),source_truth_supplied=False)
        put(target/(job['key']+'.json'),result);output.append(desc(target/(job['key']+'.json')))
    assert not blocked;put(target/'completion.json',{'status':'COMPLETE','outputs':output,'counts':ledger.counts,'blocked_reads':blocked,'neural_calls':0})
def audit(reg):
    tx=json.loads((RUN/'tx/completion.json').read_text());rx=json.loads((RUN/'rx/completion.json').read_text());assert tx['counts']=={'transmit':24} and rx['counts']['header']==24 and not rx['blocked_reads']
    outputs=[];qualified=set()
    for item,t,d in zip(reg['frames'],tx['outputs'],rx['outputs']):
        verify(d);r=json.loads(pathlib.Path(d['path']).read_text());assert item['key']==t['key']==r['key'] and not r['source_truth_supplied']
        assert r['observation_sha256']==t['observed_sha256'];p=r['profile'];b=r['body'] or {}
        exact=bool(p and p['profile_id']==t['transmission']['profile_id'] and b.get('hard_bits_sha256')==t['transmission']['information_sha256'])
        accepted=r['status'] in ['PAYLOAD_PARSED','NULL_PROFILE']
        if accepted and exact:
            assert hashlib.sha256(bytes.fromhex(b['payload_hex'])).hexdigest()==t['payload_sha256']
            if item['purpose']=='19dB_selected_MCS_connectivity':qualified.add((item['q'],item['nominal_rate']))
        outputs.append({'key':item['key'],'snr_db':item['snr_db'],'q':item['q'],'rate':item['nominal_rate'],'endpoint':item['endpoint'],
          'purpose':item['purpose'],'actual_status':r['status'],'accepted':accepted,'exact_post_exit_audit':exact,'accepted_wrong_information':accepted and not exact,
          'header_symbols':68,'body_symbols':956,'N':1024,'source_bits':t['transmission']['payload_bits'],'frame_energy':t['transmission']['energy']})
    wanted={(f['q'],f['nominal_rate']) for f in reg['frames'] if f['purpose']=='19dB_selected_MCS_connectivity'}
    assert qualified==wanted,(qualified,wanted)
    null=outputs[-1];assert null['purpose']=='explicit_null_full_frame' and null['source_bits']==0 and null['N']==1024
    put(OUT/'packet_summary.json',outputs)
    put(OUT/'completion.json',{'status':'PASS_24_ENGINEERING_FRAMES','registration_sha256':desc(REG)['sha256'],'frames':24,'header_decodes':24,'body_decodes':rx['counts']['body'],
      'neural_calls':0,'quality_scores':0,'training_updates':0,'engineering_wireless_frames':24,'calibration_logical_frames':0,'evaluation_frames':0,
      'exact_selected_MCS_connectivity':sorted([list(x) for x in qualified]),'accepted_wrong_information':sum(x['accepted_wrong_information'] for x in outputs),
      'failure_outputs_retained':True,'TX_truth_only_after_RX_exit':True,'outputs':[desc(OUT/'packet_summary.json')],'ended_unix':time.time()})
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['tx','rx','audit']);p.add_argument('--expected-registration');a=p.parse_args()
    # Independent RX never opens a TX/source registration, even for metadata.
    if a.role=='rx':assert a.expected_registration is None;rx();return
    assert desc(REG)['sha256']==a.expected_registration;reg=json.loads(REG.read_text())
    for d in reg['source_files']:verify(d)
    if a.role=='tx':tx(reg)
    else:audit(reg)
if __name__=='__main__':main()
