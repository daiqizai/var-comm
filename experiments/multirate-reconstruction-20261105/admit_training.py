"""First main launch only. Re-admission and resume require a new explicit receipt."""
from pathlib import Path
import json,os,subprocess,sys,time
from runtime import L,OLD,R,PY,env,sha,write
def main():
    S=Path(__file__).resolve().parent;root=L/'controllers';out=L/'training_v1'
    assert (OLD/'STOP_P2').exists() and not (L/'STOP_MAIN').exists()
    assert not out.exists() and not (root/'training_v1').exists()
    for role in ['qualification_v3','qualification_rx_v1']:
        actual=json.loads((root/role/'actual_wait.json').read_text());assert actual['returncode']==0 and actual['child_gone']
    import torch
    q=L/'qualification_v3';rx=L/'qualification_rx_v1'
    assert torch.equal(torch.load(q/'tx_received_ids.pt',weights_only=True),torch.load(rx/'received_ids.pt',weights_only=True))
    a=torch.load(q/'inprocess_rx.pt',weights_only=True);b=torch.load(rx/'rx.pt',weights_only=True)
    assert torch.equal(a,b),'Separate process pixels differ'
    assert json.loads((q/'complete.json').read_text())['passed']
    gpus=[1,2,4,5]
    gpu_rows=[]
    for gpu in gpus:
        row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        parts=[s.strip() for s in row.split(',')];assert int(parts[3])<100 and int(parts[4])<5,row;gpu_rows.append(row)
    out.mkdir();corpus=L/'corpus_coco116k_v2/bootstrap_train_manifest.json';assert corpus.exists()
    # Use all verified downloads already available, instead of waiting for the
    # entire corpus. Each later data transition creates an immutable manifest.
    from train import Corpus,available_corpus
    data=Corpus(corpus,20261105);corpus=Path(available_corpus(out,data))
    bindings={str(S/n):sha(S/n) for n in ['runtime.py','model.py','codec.py','train.py','admit_training.py']}
    request=dict(output=str(out),seed=20261105,training_manifest=str(corpus),training_manifest_sha256=sha(corpus),
        schedule=[dict(resolution=256,updates=100000),dict(resolution=512,updates=50000)],
        global_batch=32,microbatch_256=4,microbatch_512=2,learning_rate=1e-5,warmup_steps=500,
        checkpoint_steps=100,wall_seconds=39600,owner_max_seconds=43200,initial_allocation_max_gpu_device_hours=48,
        gpus=gpus,devices=gpu_rows,bindings=bindings,bootstrap_scope='actual growing verified training corpus; no claim of completed large-scale training',
        actual_packet_protocol='17 bytes paid mode, prefix endpoint, stream lengths and CRC; top-k rANS; source packet, not a new radio frame',
        precision='BF16 E/D matrix operations, FP32 parameter/optimizer/entropy/CDF; TF32 off',
        reconstruction='one deterministic conditional transformer route',
        losses='MSE + LPIPS-VGG + 1.0 soft bpp + 4.0 budget excess bpp + VQ/commitment',
        full_representation_frequency='one of four optimizer updates',
        data_transition='only at checkpoints, >=1000 new verified unique training images; exclude heldout hashes',
        old_P2='PAUSED_BY_USER_INCOMPLETE; never resumed')
    rp=out/'request.json';write(rp,request)
    write(out/'admission.json',dict(passed=True,independent_rx_indices_exact=True,independent_rx_pixels_exact=True,
        qualifications=[str(q),str(rx)],gpu_inventory=gpu_rows,old_stop=True,training_manifest_sha256=sha(corpus),registered=time.time()))
    owner=L/'training_owner_v1.py';assert not owner.exists()
    body='import sys\nsys.path.insert(0,'+repr(str(S))+')\nfrom runtime import *\nsys.path.insert(0,str(R/"experiments/rdvq-prefix-20261104"))\nfrom owned_process import run_role\ne=env();e["CUDA_VISIBLE_DEVICES"]="1,2,4,5"\n'
    command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(S/'train.py'),'--request',str(rp)]
    body+='run_role('+repr(str(root))+',"training_v1",'+repr(command)+',e,list(range(48,80)),[1,2,4,5],43200,'+repr({**bindings,str(rp):sha(rp)})+')\n'
    owner.write_text(body)
    p=subprocess.Popen([PY,str(owner)],stdin=subprocess.DEVNULL,stdout=(L/'training_owner_v1.stdout').open('xb'),stderr=(L/'training_owner_v1.stderr').open('xb'),start_new_session=True,env=env())
    write(out/'launch.json',dict(owner_pid=p.pid,request_sha256=sha(rp),time=time.time()));print(json.dumps(dict(owner_pid=p.pid,request=str(rp),gpus=gpus)))
if __name__=='__main__':main()
