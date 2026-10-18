"""Post-exit accounting of independent RX results; no truth-based correction."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
import control as c
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True);a=p.parse_args();assert c.desc(c.REG)['sha256']==a.expected_registration
    reg=json.loads(c.REG.read_text());c.success(c.RUN/'rx.actual_wait.json')
    done=json.loads((c.RUN/'rx/completion.json').read_text());assert done['status']=='COMPLETE' and not done['blocked_reads']
    counts={};calls=c.RUN/'rx/calls'
    if reg['new_jobs']:
        for f in calls.glob('*.reserved.json'):
            x=json.loads(f.read_text());counts[x['kind']]=counts.get(x['kind'],0)+1
            assert f.with_name(f.name.replace('.reserved.','.complete.')).exists()
        assert not list(calls.glob('*.failed.json'))
        assert counts==done['counts'] and counts['model_load']==1 and counts['rx_prefix']==reg['new_jobs']
    else:assert not calls.exists() and done['counts']=={}
    reused=json.loads((c.OUT/'reused_states.json').read_text());jobs=json.loads((c.RUN/'wire_inputs/jobs.json').read_text());jobmap={j['key']:j for j in jobs}
    assert len(done['outputs'])==reg['new_jobs'] and len(reused)==reg['reused_jobs']
    states=[]
    for s in reused:states.append(dict(s,status='IMAGE_READY'))
    for r in done['outputs']:
        j=jobmap[r['key']];assert j['paid_profile']==r['endpoint']
        if r.get('status')=='SOURCE_PARSE_FAIL':
            states.append({'key':r['key'],'endpoint':r['endpoint'],'status':'SOURCE_PARSE_FAIL','image':None,'source_error':r['source_error'],'input':j['input'],'origin':'NEW_ACTUAL_RX'})
        else:
            assert pathlib.Path(r['path'])==c.RUN/'rx'/(r['key']+'.npz')
            states.append({'key':r['key'],'endpoint':r['endpoint'],'status':'IMAGE_READY','image':c.desc(r['path']),'input':j['input'],'origin':'NEW_ACTUAL_RX'})
    assert len({s['key'] for s in states})==len(states)==reg['actual_unique_jobs']
    successful=0
    for s in states:
        c.verify(s['input'])
        if s['image'] is not None:
            c.verify(s['image'])
            with np.load(s['image']['path'],allow_pickle=False) as z:
                x=z['float_output'];assert x.dtype==np.float32 and x.shape==(3,256,256) and np.isfinite(x).all() and (x>=0).all() and (x<=1).all()
                s['float_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
            successful+=s['origin']=='NEW_ACTUAL_RX'
    assert counts.get('rx_render',0)==successful
    bykey={s['key']:s for s in states};logical=json.loads((c.O/'a2b_link_v1/logical_frames.json').read_text());assert len(logical)==18000
    rows=[]
    for r in logical:
        state=bykey[r['received_source_state']] if r['received_source_state'] else None
        row=dict(r);row.update(receiver_status='FIXED_GRAY' if state is None else state['status'],image=None if state is None else state['image'],image_float_sha256=None if state is None else state.get('float_sha256'))
        rows.append(row)
    c.put(c.OUT/'actual_states.json',states);c.put(c.OUT/'logical_receivers.json',rows)
    c.put(c.OUT/'completion.json',{'status':'ACTUAL_SOURCE_RX_CLOSED_READY_FOR_METRICS','registration_sha256':a.expected_registration,'logical_frames':18000,
      'new_prefix_calls':counts.get('rx_prefix',0),'new_image_renders':successful,'reused_states':len(reused),'actual_unique_states':len(states),'source_parse_failures':sum(s['status']=='SOURCE_PARSE_FAIL' for s in states),
      'fixed_gray_logical_frames':sum(r['image'] is None for r in rows),'unresolved':0,'wireless_frames':0,'scores':0,'truth_correction':False,
      'outputs':[c.desc(c.OUT/'actual_states.json'),c.desc(c.OUT/'logical_receivers.json')]})
if __name__=='__main__':main()
