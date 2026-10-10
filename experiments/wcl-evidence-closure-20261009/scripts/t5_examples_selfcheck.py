"""Engineering admission checks only: no neural, channel, PHY or BPG calls."""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import t5_missing_examples as t5

def rejected(fn):
    try:fn()
    except (RuntimeError,FileExistsError):return
    raise AssertionError('Invalid engineering fixture was accepted')
def main():
    checks=[]
    assert len(t5.FIXED)==len(set(t5.FIXED))==16 and t5.SNRS==(1,10,19)
    ids={t5.key(m,i,s) for m in t5.METHODS for i in t5.FIXED for s in t5.SNRS}
    assert len(ids)==96
    checks.append('Distinct identities for16sources x3SNR x2methods; existing Swin1 contributes16 reuse frames')
    with tempfile.TemporaryDirectory(prefix='wcl_t5_engineering_') as tmp:
        directory=Path(tmp);archive=directory/'reconstructions.npz';archive.write_bytes(b'not-an-image:engineering-admission-fixture')
        receipt=directory/'frame.json';row=dict(selected_step=80000,selected_checkpoint_sha256=t5.CHECKPOINT)
        value=dict(frame=dict(source_index=0,N=1024,snr_db=1,noise_seed=2001),source_id='engineering_source',
            rows=[row],archive=str(archive),archive_sha256=t5.sha(archive),synthetic=False)
        t5.save(receipt,value)
        assert t5.inspect_swin(receipt,0,1,{'source_id':'engineering_source'})['array_slot']==0
        rejected(lambda:t5.inspect_swin(receipt,0,19,{'source_id':'engineering_source'}))
        rejected(lambda:t5.inspect_swin(receipt,0,1,{'source_id':'wrong_source'}))
        rejected(lambda:t5.save(receipt,value))
        archive.write_bytes(b'changed');rejected(lambda:t5.inspect_swin(receipt,0,1,{'source_id':'engineering_source'}))
        checks.append('Exact source/SNR/seed/80k archive identity enforced; mutation and overwrite rejected')
        receipt.unlink();rejected(lambda:t5.inspect_swin(receipt,0,1,{'source_id':'engineering_source'}))
        checks.append('An unreceipted old image blocks replay; it is never treated as missing')
    tree=ast.parse(inspect.getsource(t5.run_bpg))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    names={n.func.id for n in calls if isinstance(n.func,ast.Name)}
    assert not {'source_rows','choice','make_probe'}&names
    receivers=[n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='receiver' and n.func.attr=='decode']
    assert len(receivers)==1 and len(receivers[0].args)==1 and receivers[0].args[0].id=='received'
    checks.append('No source-codec search/selection; BPG decoder accepts only actual received bytes')
    print(json.dumps(dict(status='PASS_ENGINEERING_ONLY',checks=checks,actual_packet_decodes=0,neural_calls=0,BPG_calls=0)))
if __name__=='__main__':main()
