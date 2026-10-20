"""Finite sensitivity RX: receives only closed RX token cache and paid wires."""
from pathlib import Path
import sys
from adapter import native_program, generation_function, temperatures

TAIL = r'''
    result=[]
    jobs=json.loads((pathlib.Path(cfg['run'])/'wire_inputs/jobs.json').read_text())
    assert len(jobs)==cfg['unique_prefixes']==296
    source=pathlib.Path(cfg['upstream'])/'compression/reciever.py'
    assert file_sha(source)==cfg['receiver_source_sha256']
    adapters={}
    for setting in ['greedy','low_tau025']:
        generation,adapter_sha=generation_function(receiver,setting,source.read_text())
        adapters[setting]=adapter_sha
        for n,job in enumerate(jobs):
            key=setting+'_'+job['key'];e=job['paid_profile']
            assert 1<=e<7 and job['generation_seed']==0
            pin(job['wire']);pin(job['prefix'])
            caption,trans,modes=unpack(pathlib.Path(job['wire']['path']).read_bytes(),e)
            assert len(trans)==len(modes)==e
            with np.load(job['prefix']['path'],allow_pickle=False) as z:
                assert set(z.files)=={f's{i+1}' for i in range(e)}
                tokens=[]
                for i in range(e):
                    t=z[f's{i+1}'].copy()
                    assert t.shape==(1,SIDES[i]**2,16) and str(t.dtype)=='int32'
                    assert np.isin(t,[0,1]).all()
                    tokens.append(torch.from_numpy(t).cuda())
            before=[digest(t.cpu().numpy().tobytes()) for t in tokens]
            image=render(tokens,e,key,'rx_render')
            assert before==[digest(t.cpu().numpy().tobytes()) for t in tokens]
            result.append({'key':job['key'],'setting':setting,'endpoint':e,
                'wire':job['wire'],'received_prefix':job['prefix'],
                'prefix_unchanged':True,**image})
            if (n+1)%25==0:print(json.dumps({'setting':setting,'completed':n+1}),flush=True)
    assert ledger.counts=={'model_load':1,'rx_render':592}
    assert not blocked
    put(out/'completion.json',{'status':'COMPLETE','counts':ledger.counts,'outputs':result,
        'blocked_reads':blocked,'private_generation_AST':adapters,'wireless_frames':0,
        'source_decodes':0,'source_truth_reads':0,'prefix_cache_reuse':592,
        'max_allocated_bytes':torch.cuda.max_memory_allocated(),
        'max_reserved_bytes':torch.cuda.max_memory_reserved(),'ended_unix':time.time()})

if __name__=='__main__':main()
'''

if __name__=='__main__':
    original=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019/a1_native.py')
    sys.path.insert(0,str(original.parent))
    exec(compile(native_program(original.read_bytes())+TAIL,str(original)+'::sensitivity','exec'),
         dict(__name__='__main__',__file__=str(original),generation_function=generation_function,temperatures=temperatures))
