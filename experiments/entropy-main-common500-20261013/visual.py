"""Independent entropy RX and frozen VAR/Dc, reuse only exact actual RX inputs."""
import argparse
import time
import traceback
import numpy as np
from common import *

def run(path,parent):
    r=registration(path);out=ART/'visual';out.mkdir();guard=guard_for(r,out,parent);device,g=resource_device(r,out);last=0.;ledger=None;b=None
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>5:
            snap,_=device.resource_snapshot(g.helper());last=time.monotonic();status(out/'resource.json',snap)
    caps=dict(model_load=1,encoder=0,source_tx=0,source_rx=4500,var_render=4500,prior_scale=90000,decoder_forward=4500)
    try:
        require(read(ART/'phy/actual_wait.json')['returncode']==0,'PHY actual exit0 required')
        pc=read(ART/'phy/completion.json');pins=read(checked(pc['frames']));require(len(pins)==4500,'Full4500 actual observations required')
        ledger=Ledger(out/'calls.sqlite',caps,guard);write(out/'launch.json',dict(pid=os.getpid(),registration=descriptor(path)))
        b,module,resolver,environment,_,g=backend(r,out,ledger,boundary)
        import ep_source_codec as partial
        import h800_ep48_link_core_v1 as link
        import h800_ep_pilot_core_v1 as pilot
        codec=partial.PartialSourceCodec(ROOT);old=read(checked(r['visual_request']));link.bind_static(codec,old['static'])
        source_driver=resolver.path(g.OLD+'experiments/content-real-64qam-20261006/h_source_driver.py')
        link.bind_independent_provider(codec,b,module,source_driver,g)
        identity=dict(models=dict(g.MODELS),source_codec=partial.partial_source_model_id(),native=r['native_identity'],FP32=True,TF32=False)
        write(out/'visual_identity.json',identity);cache={};rows=[];started=time.monotonic();token_cache={}
        for index,pin in enumerate(pins):
            boundary();frame=read(checked(pin));require(frame['frame_index']==index,'Fixed logical order')
            actual=frame['actual_RX'];key=digest(pilot.source_input(actual,identity));i=frame['source_index'];b.source_index=i
            if key not in cache:
                image,evidence=link.recover_and_render(actual,codec,b,partial,ledger,boundary)
                ip=out/'images'/f'{len(cache):05d}.npz';ip.parent.mkdir(exist_ok=True)
                with ip.open('xb') as f:np.savez(f,image=image)
                cache[key]=dict(image_archive=descriptor(ip),image_sha256=array_sha(image),actual_RX_input_key=key,evidence=evidence)
                mode='ACTUAL_NEW'
            else:mode='EXACT_ACTUAL_RX_INPUT_REUSE'
            rec=cache[key];ev=rec['evidence'];rxp=actual['rx_profile'];body=actual['body']
            # Post-decoding consistency audit, never an input to RX or rendering.
            # An intact same-host bitstream failing source decoding is engineering failure.
            matched=rxp is not None and rxp['profile_id']==frame['TX_profile']['profile_id'] and body is not None and body['crc_accepted'] and body['parser_accepted'] and array_sha(np.asarray(body['payload'],dtype=np.uint8))==frame['payload_sha256']
            if matched:
                require(ev['kind']=='tokens','Intact registered stream failed independent source decode')
                if i not in token_cache:
                    with np.load(checked(r['sources'][i]['archive']),allow_pickle=False) as z:token_cache={i:z['tokens'].copy()}
                require(np.array_equal(np.asarray(ev['received_tokens']),token_cache[i][:partial.token_count(rxp['m'],rxp['K'])]),'Intact same-host decoded token mismatch; stop, not a channel failure')
            row={k:frame[k] for k in ('frame_index','method','source_index','source_id','snr_db','noise_seed','N','public_frame_counter','policy','actual_m','actual_K','actual_T','arithmetic_bits','capacity_bits')}
            row.update(physical_frame=pin,reconstruction=rec,reuse=mode,physical_status=actual['status'],
                source_status=ev['source_status'] if ev['source_decode_called'] else actual['status'],
                received_m=None if rxp is None else rxp['m'],received_K=None if rxp is None else rxp['K'],header_ok=actual['header']['header_ok'],
                crc_accepted=None if body is None else body['crc_accepted'],parser_accepted=None if body is None else body['parser_accepted'],
                fixed_gray=ev['kind']=='gray',post_decode_intact_stream_audit=bool(matched),source_truth_supplied_to_RX=False)
            rp=out/'logical'/f'{index:04d}.json';write(rp,row);rows.append(descriptor(rp))
            if (index+1)%90==0:status(OUT/'visual_status.json',dict(completed_frames=index+1,total_frames=4500,unique_recoveries=len(cache),counts=ledger.summary(),elapsed_seconds=time.monotonic()-started))
        b.close();resolver.reverify();environment.reverify();counts=ledger.summary();require(counts['unresolved']==0 and len(rows)==4500,'Full receiver closure')
        write(out/'frames.json',rows);write(out/'completion.json',dict(status='COMPLETE_VISUAL4500',logical_frames=4500,actual_unique_RX_inputs=len(cache),
            exact_input_reuses=4500-len(cache),frames=descriptor(out/'frames.json'),counts=counts,visual_identity=identity,source_truth_supplied_to_RX=False));ledger.close()
    except BaseException as e:
        if b is not None:b.failure_evidence()
        write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--parent',type=int,required=True);a=p.parse_args();run(a.registration,a.parent)
