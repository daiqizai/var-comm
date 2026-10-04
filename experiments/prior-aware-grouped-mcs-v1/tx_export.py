"""Export label-free TX tokens only after final1000 policies and BLER freeze."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import time
import numpy as np
import source_quality as q
import quality_driver as driver


def validate_archive(path, required_ms, *, tokens=None):
    expected={'tokens',*[f'entropy_order_m{m}' for m in required_ms]}
    with np.load(path,allow_pickle=False) as archive:
        q.require(set(archive.files)==expected,'TX archive contains unregistered fields, labels or missing orders')
        actual=archive['tokens'].copy();q.split_tokens(actual)
        if tokens is not None:q.require(np.array_equal(actual,tokens),'TX archive source tokens changed')
        orders={}
        for m in required_ms:
            order=archive[f'entropy_order_m{m}'].copy()
            q.require(np.issubdtype(order.dtype,np.integer) and order.shape==(q.SIZES[m]**2,)
                      and np.array_equal(np.sort(order),np.arange(q.SIZES[m]**2)),'TX entropy order is not a full permutation')
            orders[m]=order
    return actual,orders


def export(native,admission,output,policies_path,codebook_path,*,stop_requested=lambda:False):
    """Only offline source coding; no classes, images or source labels go to RX."""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    q.require(native.loaded['identity']==admission['visual_identity'],'Final calibration visual identity changed')
    policies=admission['policies'];required_ms=sorted(set(admission['required_ms']))
    q.require(all(type(m)is int and 4<=m<=9 for m in required_ms),'Unsupported partial source scale')
    # This is the first development access. The caller must call admit_frozen
    # before native construction/export; its final1000 and refinement gates are
    # repeated here rather than accepting an arbitrary development manifest.
    q.require(policies['status']=='CALIBRATION_POLICIES_SELECTED' and policies['source_count']==1000
              and policies['stage']=='final' and policies['synthetic'] is False and policies['development_read'] is False,
              'Final complete calibration freeze required before development TX export')
    data=native.data('development');records=data['records']
    ids=[r['image_id'] for r in records]
    q.require(len(ids)==len(set(ids))==100 and set(ids).isdisjoint(policies['source_ids']),
              'Original distinct 100-source development population required')
    identity=dict(schema='uep_tx_tokens_v1',population='development',source_count=100,source_ids=ids,
        source_identity=[dict(source_index=i,source_id=r['image_id'],preprocessing_id=r['preprocessing_id']) for i,r in enumerate(records)],
        selected_policies_sha256=q.sha(policies_path),codebook_sha256=q.identity(admission['codebook']['entries']),
        codebook_file_sha256=q.sha(codebook_path),visual_identity=native.loaded['identity'],
        numerical_runtime=native.flags,entropy_rounding_digits=native.receiver.ENTROPY_DECIMALS,
        entropy_tie_rule='ascending spatial index',required_ms=required_ms,required_partial_scales=required_ms,
        generation=q.GENERATION,labels_given_to_receiver=False,source_images_exported=False,
        source_bindings=dict(native.driver_bindings,**{str(Path(__file__).resolve()):q.sha(__file__)}),
        input_bindings=admission['input_bindings'],data_bindings=data.get('bindings',{}),
        quality_selection_updates=0,training_updates=0)
    q.seal(output/'registration.json',identity);binding=q.identity(identity)
    states=[dict(state_id=q.state_id(m,1),m=m,K=1) for m in required_ms]
    sources=[];outputs={};started=time.monotonic()
    for index,record in enumerate(records):
        if stop_requested() or (output/'STOP').exists():
            q.write(output/'status.json',dict(status='PAUSED_AT_SOURCE_BOUNDARY',sources=index));native.frozen();return None
        tokens=data['T'][index].cpu().numpy().astype(np.int64);q.split_tokens(tokens)
        path=output/'tokens'/('%04d.npz'%index);path.parent.mkdir(parents=True,exist_ok=True)
        cp_path=output/'source_checkpoints'/('%04d.json'%index)
        if cp_path.exists():
            cp=q.read(cp_path)
            q.require(cp['binding']==binding and cp['payload_sha256']==q.identity({k:v for k,v in cp.items() if k!='payload_sha256'})
                      and cp['source_index']==index and cp['source_id']==record['image_id'] and cp['preprocessing_id']==record['preprocessing_id']
                      and cp['archive']==str(path) and cp['archive_sha256']==q.sha(path),'TX source checkpoint changed')
            validate_archive(path,required_ms,tokens=tokens)
        else:
            began=time.monotonic();arrays=dict(tokens=tokens)
            if required_ms:
                renderer=q.SourceRenderer(native.loaded,native.receiver,tokens,states,use_snapshots=True)
                try:
                    for m in required_ms:
                        prior,logits=renderer._open(m)
                        try:arrays[f'entropy_order_m{m}']=q.entropy_order(native.receiver,logits[0].float().cpu().numpy())
                        finally:prior.close()
                finally:renderer.close()
            temporary=path.with_name(path.name+'.tmp')
            with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
            os.replace(temporary,path);validate_archive(path,required_ms,tokens=tokens)
            cp=dict(binding=binding,source_index=index,source_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
                    archive=str(path),archive_sha256=q.sha(path),source_seconds=time.monotonic()-began)
            cp['payload_sha256']=q.identity(cp);q.seal(cp_path,cp)
        sources.append({k:cp[k] for k in ('source_index','source_id','preprocessing_id','archive','archive_sha256')})
        outputs[str(cp_path)]=q.sha(cp_path);outputs[str(path)]=cp['archive_sha256']
        q.write(output/'status.json',dict(status='RUNNING',sources=index+1,total_sources=100,elapsed_seconds=time.monotonic()-started))
    native.frozen()
    manifest=dict(identity,sources=sources,status='UEP_TX_TOKENS_COMPLETE',registration_sha256=q.sha(output/'registration.json'),outputs=outputs)
    q.seal(output/'manifest.json',manifest)
    q.write(output/'status.json',dict(status='COMPLETE',sources=100,elapsed_seconds=time.monotonic()-started))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-runtime',type=Path)
    for name in ('policies','quality-bundle','quality-completion','bler','codebook'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    from evaluate import admit_frozen
    admitted=admit_frozen(args.policies,args.quality_bundle,args.quality_completion,args.bler,args.codebook)
    with driver.single_writer(args.output):
        native=driver.build_native(args.root,args.native_runtime or args.root/'experiments/m1-n2048-full-grid-20261004')
        signal.signal(signal.SIGTERM,driver.stop);signal.signal(signal.SIGINT,driver.stop)
        export(native,admitted,args.output,args.policies,args.codebook,stop_requested=lambda:driver.boundary(native))


if __name__=='__main__':main()
