"""Four calibration-source checks with real frozen weights; no development.

This is a disposable forward-only qualification, not policy selection. The
incorrect accepted-prefix diagnostic is explicitly injected; it is not an
observed channel false acceptance or a simulated feedback mechanism.
"""
from __future__ import annotations
import argparse
import copy
import inspect
from pathlib import Path
import time

import common as c
import numpy as np
import torch
import partial_phy as phy
import partial_receiver as rx


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _same_tokens(a, b):
    return len(a) == len(b) and all(np.array_equal(x,y) for x,y in zip(a,b))


@torch.no_grad()
def qualify(loaded, data, count=4):
    if count != 4 or len(data['records']) != 1000:
        raise ValueError('Qualification is fixed to first four original calibration sources')
    started=time.time();bindings=c.source_bindings();rows=[];wrong_prefix=[]
    c.assert_frozen(loaded)
    _assert(not torch.is_autocast_enabled(), 'Qualification requires pinned FP32 without autocast')
    _assert(not torch.backends.cuda.matmul.allow_tf32, 'FP32 matmul TF32 must remain disabled')
    _assert(not torch.backends.cudnn.allow_tf32, 'FP32 convolution TF32 must remain disabled')
    _assert(torch.are_deterministic_algorithms_enabled(), 'Deterministic numerical protocol required')
    for function in [phy.receive,rx.complete_partial,rx.prefix_logits,rx.received_cache_key]:
        _assert(not ({'truth','truth_next','clean','source_id','true_action'} & set(inspect.signature(function).parameters)),
                'RX interface exposes unsent information')
    ledgers=[phy.action_record(a) for N in (512,1024) for family in phy.PHY_FAMILIES for a in phy.action_grid(N,family)]
    for rec in ledgers:
        _assert(rec['N_header']+rec['N_prefix']+rec['N_partial']==rec['N'],'Ledger does not pay full budget')
        _assert(rec['prefix_effective_rate']<=.9,'Prefix rate cap')
        if rec['q']:
            _assert(rec['partial_effective_rate']<=.9,'Partial rate cap')
            _assert(rec['mask_bits']==(rec['next_scale_tokens'] if rec['order']=='oracle' else 0),'Paid mask count')
    for i in range(count):
        c.check();record=data['records'][i];scales=c.split_tokens(data['T'][i].numpy())
        # q0 uses the new paid header but must reproduce the original completion
        # for every strong whole mode, including legal N1024/16QAM m8.
        whole_results={}
        for m in range(4,9):
            action=phy.Action(1024,'16QAM',m,0,'whole')
            wave,ledger=phy.transmit(scales,action)
            event=phy.receive(wave,60.,1024,'16QAM')
            _assert(event['header_ok'] and event['prefix_crc_ok'] and event['action']==action,'q0 noiseless PHY failed')
            _assert(_same_tokens(event['prefix'],scales[:m]),'q0 noiseless received prefix changed')
            result=rx.complete_partial(loaded['vae'],loaded['var'],event,loaded['device'],return_logits=True)
            legacy=c.legacy.complete_latent(loaded['vae'],loaded['var'],event['prefix'],1000,loaded['device']).float().cpu()
            _assert(torch.equal(result['fhat'],legacy),'q0 differs bitwise from original complete_latent')
            _assert(_same_tokens(result['tokens'][:m],event['prefix']),'Full received prefix was not fixed')
            whole_results[m]=result
            rows.append(dict(source_index=i,source_id=record['image_id'],m=m,order='whole',q=0,
                q0_bitwise_parity=True,header_ok=True,prefix_crc_ok=True,**{'E':ledger['E']}))
        for m in range(4,8):
            # Independent fresh TX/RX calls, with no shared hidden state or logits.
            tx_logit=rx.prefix_logits(loaded['vae'],loaded['var'],scales[:m],loaded['device'])
            for order in phy.ORDERS:
                maximum=phy.max_legal_q(1024,'16QAM',m,order)
                q=min(phy.next_length(m)//4,maximum)
                _assert(q>0,'Qualification partial action must be legal')
                action=phy.Action(1024,'16QAM',m,q,order)
                tx_positions=rx.order_positions(scales[:m],order,q,tx_logit,scales[m] if order=='oracle' else None)
                wave,ledger=phy.transmit(scales,action,tx_positions)
                event=phy.receive(wave,60.,1024,'16QAM')
                _assert(event['header_ok'] and event['prefix_crc_ok'] and event['partial_usable'],'Noiseless independent packets failed')
                _assert(event['action']==action and _same_tokens(event['prefix'],scales[:m]),'Independent RX header/prefix changed')
                rx_logit=rx.prefix_logits(loaded['vae'],loaded['var'],event['prefix'],loaded['device'])
                _assert(np.array_equal(tx_logit,rx_logit),'Independent endpoint FP32 logits are not bitwise reproducible')
                if order=='oracle':
                    rx_positions=event['partial_positions']
                else:
                    rx_positions=rx.order_positions(event['prefix'],event['action'].order,event['action'].q,rx_logit)
                _assert(np.array_equal(tx_positions,rx_positions),'Independent endpoint positions disagree')
                result=rx.complete_partial(loaded['vae'],loaded['var'],event,loaded['device'],return_logits=True)
                diagnostics=result['diagnostics'];positions=np.asarray(diagnostics['known_positions'],np.int64)
                _assert(diagnostics['partial_used'] and diagnostics['known_tokens_fixed'],'Received subset was not applied/fixed')
                _assert(np.array_equal(positions,rx_positions),'Completion did not derive RX positions independently')
                _assert(np.array_equal(result['tokens'][m][positions],event['partial_values']),'Actual received values changed')
                _assert(_same_tokens(result['tokens'][:m],event['prefix']),'Received full prefix changed')
                missing=np.setdiff1d(np.arange(phy.next_length(m)),positions)
                _assert(np.array_equal(result['tokens'][m][missing],result['next_prior_tokens'][missing]),
                    'Missing same-scale tokens were incorrectly conditioned on the clamped subset')
                _assert(np.array_equal(result['next_prior_tokens'],whole_results[m]['next_prior_tokens']),
                    'The prior depends on hidden TX action/partial information')
                # CRC decisions are deliberately injected here to qualify the
                # GPU receiver branch, separately from native PHY CPU tests.
                for fail in ['prefix','partial']:
                    failed=copy.deepcopy(event);failed[fail+'_crc_ok']=False
                    failed['partial_usable']=False;failed['partial_discard_reason']=fail+'_crc_failure'
                    out=rx.complete_partial(loaded['vae'],loaded['var'],failed,loaded['device'])
                    _assert(not out['diagnostics']['partial_used'],'CRC failure leaked partial data')
                    _assert(torch.equal(out['fhat'],whole_results[m]['fhat']) and _same_tokens(out['tokens'],whole_results[m]['tokens']),
                        'Discard branch differs from completion of the same actual hard prefix')
                # The wire never sends non-oracle positions. An accepted but
                # incorrect prefix can therefore desynchronise deployable order.
                if order in ['entropy','random']:
                    altered=[x.copy() for x in event['prefix']];altered[0][0]^=1
                    altered_logits=rx.prefix_logits(loaded['vae'],loaded['var'],altered,loaded['device'])
                    altered_positions=rx.order_positions(altered,order,q,altered_logits)
                    wrong_prefix.append(dict(source_index=i,m=m,q=q,order=order,
                        diagnostic_kind='injected_wrong_prefix_with_accepted_crc_flag',
                        same_positions=bool(np.array_equal(tx_positions,altered_positions)),
                        positional_agreement_fraction=float(np.mean(tx_positions==altered_positions)),
                        no_feedback_or_receiver_sort_switch=True))
                rows.append(dict(source_index=i,source_id=record['image_id'],m=m,q=q,order=order,
                    tx_rx_logits_bitwise_equal=True,tx_rx_order_equal=True,known_tokens_fixed=True,
                    missing_same_scale_prior_unchanged=True,crc_failure_discard_equal=True,E=ledger['E']))
            c.status('m1_qualification',sources=i+1,total=count,m=m,completed_checks=len(rows))
    header_erasure=dict(header_ok=False)
    erased=rx.complete_partial(loaded['vae'],loaded['var'],header_erasure,loaded['device'])
    _assert(erased['fhat'] is None and erased['tokens']==[] and not erased['diagnostics']['decoder_applied'],
            'Header erasure invented a latent or applied a decoder')
    c.assert_frozen(loaded);c.verify_bindings(bindings)
    return dict(status='REAL_WEIGHT_QUALIFICATION_PASS',calibration_sources=count,development_read=False,
        source_ids=[r['image_id'] for r in data['records'][:count]],
        preprocessing_ids=[r['preprocessing_id'] for r in data['records'][:count]],
        frozen_model_identity=loaded['identity'],qualification_source_bindings=bindings,
        calibration_bindings=data['bindings'],training_updates=0,cache_used=False,
        endpoint_logits_bitwise_equal=True,q0_bitwise_old_completion_parity=True,
        checks=rows,grid_ledgers=ledgers,wrong_accepted_prefix_diagnostic=wrong_prefix,
        wrong_prefix_scope='Injected diagnostic only; does not establish an observed channel CRC false-accept rate',
        duration_seconds=time.time()-started)


def main(count=4):
    c.check();loaded=c.setup();data=c.assets.load_calibration()
    with torch.inference_mode():report=qualify(loaded,data,count)
    path=c.RESULT/'m1_qualification.json';c.write(path,report)
    receipt=dict(status='REAL_WEIGHT_QUALIFICATION_PASS',qualification_sha256=c.sha(path),
        qualification_path=str(path),calibration_sources=count,development_read=False,
        training_updates=0,source_bindings=report['qualification_source_bindings'])
    c.write(c.OUT/'qualification.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sources',type=int,default=4)
    args=parser.parse_args();main(args.sources)
