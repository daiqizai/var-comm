"""Independent physical TX/RX and post-exit accounting for <=18000 logical frames."""
import argparse,hashlib,json,os,pathlib,sys,time,traceback
import numpy as np
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E/'phy_v1'));sys.path.insert(0,str(E))
import link
from a1_owner import R,L,O,desc,verify,put
from plan import events,digest
RUN=L/'a2b_link_v1';OUT=O/'a2b_link_v1';REG=OUT/'registration.json'
class Ledger:
    def __init__(self,root,caps):
        self.f=(root/'calls.jsonl').open('xb');self.caps=caps;self.counts={};self.sequence=0
    def record(self,row):
        self.f.write((json.dumps(row,sort_keys=True)+'\n').encode());self.f.flush();os.fsync(self.f.fileno())
    def call(self,kind,key,fn):
        n=self.counts.get(kind,0)+1;assert n<=self.caps[kind];self.counts[kind]=n;self.sequence+=1;i=self.sequence;t=time.time()
        self.record({'event':i,'state':'RESERVED','kind':kind,'key':key,'started_unix':t})
        try:
            value=fn();self.record({'event':i,'state':'COMPLETE','seconds':time.time()-t});return value
        except BaseException:self.record({'event':i,'state':'FAILED','error':traceback.format_exc(),'seconds':time.time()-t});raise
def cache_env(target):
    cache=target/'runtime_cache';cache.mkdir();(cache/'tmp').mkdir();os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
    import tempfile;tempfile.tempdir=str(cache/'tmp')
def tx(reg):
    target=RUN/'tx';target.mkdir();incoming=RUN/'received_inputs';incoming.mkdir();cache_env(target)
    rt=link.Runtime();assert rt.backend.identity==reg['backend_identity'];ledger=Ledger(target,{'encode_waveform':18000})
    sources=json.loads(pathlib.Path(reg['streams']['path']).read_text())['sources'];assert len(sources)==1000
    decisions=json.loads(pathlib.Path(reg['nomination']['path']).read_text())['decisions'];frames=events(sources,decisions,link.templates());assert len(frames)==18000
    put(target/'frame_plan.json',frames);put(incoming/'catalogue.json',rt.catalogue);put(incoming/'public.json',{'count':18000,'header_cap':18000,'body_cap':18000,'session':link.SESSION,'generation_seed':0})
    jobs=[];outputs=[];waves={};previous_source=None
    for index,f in enumerate(frames):
        if f['source_index']!=previous_source:waves={};previous_source=f['source_index']
        wavekey=f['waveform_key']
        if wavekey not in waves:
            if f['wire'] is None:payload=b''
            else:verify(f['wire']);payload=pathlib.Path(f['wire']['path']).read_bytes()
            frame,receipt=ledger.call('encode_waveform',wavekey,lambda:rt.transmit(payload,f['profile_id'],f['counter']))
            waves[wavekey]=(frame,receipt,hashlib.sha256(payload).hexdigest())
        frame,receipt,payloadsha=waves[wavekey]
        noise=link.ep.standard_noise(f['source_id'],f['noise_seed'])*10**(-f['snr_db']/20);observed=frame+noise
        path=incoming/(f['key']+'.npy')
        with path.open('xb') as stream:np.save(stream,observed,allow_pickle=False)
        jobs.append({'key':f['key'],'snr_db':f['snr_db'],'counter':f['counter'],'observation':desc(path)})
        outputs.append({'key':f['key'],'transmission':receipt,'payload_sha256':payloadsha,'noise_sha256':link.ep.array_sha(noise),'observation_sha256':link.ep.array_sha(observed)})
        if (index+1)%300==0:print(json.dumps({'stage':'TX','logical_observations':index+1,'encoded_waveforms':ledger.counts}),flush=True)
    put(incoming/'jobs.json',jobs);put(target/'completion.json',{'status':'COMPLETE','logical_frames':len(frames),'counts':ledger.counts,'outputs':outputs,'neural_calls':0})
def rx():
    target=RUN/'rx';target.mkdir();cache_env(target)
    from a1_native import install_rx_boundary
    rtroot=R.parent.parent/'var_comm_runtime_20261010'
    allowed=[E,link.OLD,R/'experiments/wcl-evidence-closure-20261009/scripts',R/'experiments/prior-aware-grouped-mcs-v1',
      R/'experiments/main-raw64-20261007',R/'src',R/'outputs/build',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',
      rtroot/'envs/frozen_ldpc_py311',rtroot/'envs/frozen_python_base_v1/lib/python3.11',rtroot/'envs/frozen_python_base_v1/lib/python311.zip',
      rtroot/'assets/frozen_namespace_v1/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/lib/python3.11/site-packages',RUN/'received_inputs',target]
    blocked=install_rx_boundary(allowed);rt=link.Runtime();assert rt.catalogue==json.loads((RUN/'received_inputs/catalogue.json').read_text())
    sys.path.insert(0,str(E/'upstream'));import wire
    public=json.loads((RUN/'received_inputs/public.json').read_text());assert public['count']==18000 and public['session']==link.SESSION
    jobs=json.loads((RUN/'received_inputs/jobs.json').read_text());assert len(jobs)==18000
    ledger=Ledger(target,{'header':18000,'body':18000});(target/'packets').mkdir();(target/'payloads').mkdir();outputs=[];unique={}
    for index,j in enumerate(jobs):
        verify(j['observation']);y=np.load(j['observation']['path'],allow_pickle=False)
        result=rt.receive(y,j['snr_db'],j['counter'],ledger,j['key']);body=result['body']
        if body is not None:
            payload=body.pop('payload')
            if payload is not None:
                body['actual_payload_sha256']=hashlib.sha256(payload).hexdigest();body['actual_payload_bytes']=len(payload)
            if result['status']=='PAYLOAD_PARSED':
                try:
                    caption,blocks,modes=wire.unpack(payload,result['profile']['endpoint'])
                    result['source_syntax']='PASS';state=digest([body['actual_payload_sha256'],result['profile']['endpoint'],0,link.SESSION])
                    if state not in unique:
                        path=target/'payloads'/(state+'.arc')
                        with path.open('xb') as stream:stream.write(payload)
                        unique[state]={'key':state,'path':str(path),'input':desc(path),'paid_profile':result['profile']['endpoint'],'generation_seed':0}
                    result['received_source_state']=state
                except wire.SourceParseError as error:result.update(status='SOURCE_PARSE_FAIL',source_error=str(error))
        result.update(key=j['key'],observation_sha256=link.ep.array_sha(y),source_truth_supplied=False)
        path=target/'packets'/(j['key']+'.json');put(path,result);outputs.append(desc(path))
        if (index+1)%300==0:print(json.dumps({'stage':'RX','logical_frames':index+1,'counts':ledger.counts,'actual_unique_payloads':len(unique)}),flush=True)
    assert not blocked and ledger.counts['header']==18000
    put(target/'actual_source_jobs.json',list(unique.values()))
    put(target/'completion.json',{'status':'COMPLETE_ACTUAL_PHY_RX','logical_frames':18000,'counts':ledger.counts,'outputs':outputs,'blocked_reads':blocked,'neural_calls':0,'actual_unique_payloads':len(unique),'jobs':desc(target/'actual_source_jobs.json')})
def ledger_closed(path,expected):
    pending={};counts={}
    for line in path.read_text().splitlines():
        a=json.loads(line);i=a['event']
        if a['state']=='RESERVED':assert i not in pending;pending[i]=a;counts[a['kind']]=counts.get(a['kind'],0)+1
        else:assert a['state']=='COMPLETE' and i in pending;del pending[i]
    assert not pending and counts==expected;return counts
def audit(reg):
    tx=json.loads((RUN/'tx/completion.json').read_text());rx=json.loads((RUN/'rx/completion.json').read_text());frames=json.loads((RUN/'tx/frame_plan.json').read_text())
    assert tx['logical_frames']==rx['logical_frames']==len(frames)==18000 and not rx['blocked_reads']
    ledger_closed(RUN/'tx/calls.jsonl',tx['counts']);ledger_closed(RUN/'rx/calls.jsonl',rx['counts']);rows=[]
    for f,t,d in zip(frames,tx['outputs'],rx['outputs']):
        verify(d);r=json.loads(pathlib.Path(d['path']).read_text());assert f['key']==r['key']==t['key']
        assert r['observation_sha256']==t['observation_sha256'] and not r['source_truth_supplied'];p=r['profile'];b=r['body'] or {}
        exact=bool(p and p['profile_id']==t['transmission']['profile_id'] and b.get('hard_bits_sha256')==t['transmission']['information_sha256'])
        accepted=r['status']=='PAYLOAD_PARSED'
        if accepted and exact:assert b['actual_payload_sha256']==t['payload_sha256']
        rows.append({k:f[k] for k in ['key','source_id','source_index','snr_db','candidate_id','noise_seed','target_endpoint','endpoint','source_no_fit']})
        rows[-1].update(actual_status=r['status'],actual_rx_endpoint=p['endpoint'] if p else None,received_source_state=r.get('received_source_state'),accepted_wrong_information=accepted and not exact,exact_post_exit_audit=exact)
    put(OUT/'logical_frames.json',rows)
    put(OUT/'completion.json',{'status':'CAL1000_PHYSICAL_FRAMES_CLOSED_AWAITING_ACTUAL_SOURCE_RX','registration_sha256':desc(REG)['sha256'],'logical_frames':18000,
      'encoded_waveforms':tx['counts']['encode_waveform'],'header_decodes':rx['counts']['header'],'body_decodes':rx['counts'].get('body',0),'unresolved':0,'new_neural_calls':0,
      'accepted_wrong_information':sum(r['accepted_wrong_information'] for r in rows),'actual_unique_payloads':rx['actual_unique_payloads'],
      'outputs':[desc(OUT/'logical_frames.json'),rx['jobs']],'ended_unix':time.time()})
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['tx','rx','audit']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='rx':assert a.expected_registration is None;rx();return
    assert desc(REG)['sha256']==a.expected_registration;reg=json.loads(REG.read_text())
    for d in reg['source_files']:verify(d)
    if a.role=='tx':tx(reg)
    else:audit(reg)
if __name__=='__main__':main()
