"""CPU proof/cache, native API and receiver-boundary tests (no model loading)."""
import ast
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import external_eval_common as c
import own_controls_cost_common as k
import own_controls_cost as runner


RUNTIME=dict(uuid='GPU-fixed',name='test-GPU',driver='test',torch='original',cuda='original')
HIFI=dict(header_accepted=True,NFE=50,t_start=50,diagnostic_probe=False,complete_author_schedule=True,
    model_parameters_unchanged=True,parameter_gradients_accumulated=False,schedule_mode='actual_data_cbr')


def sample(i=0):
    rgb=np.zeros((3,256,256),np.float32)+np.float32(i/255)
    row=dict(image_id='cal-'+str(i),class_index=i,
             preprocessing_id=__import__('hashlib').sha256(np.rint(rgb*255).astype(np.uint8).tobytes()).hexdigest())
    return dict(k.source_record(i,row,rgb),rgb=rgb)


def make_frame(directory,binding,spec,source):
    wave=np.ones((spec['N'],2),np.float32);row=runner.row_base(spec,source,RUNTIME)
    row.update(RX_seconds=.01*(1+spec['source_index']),header_accepted=True)
    if spec['method']==c.METHODS[1]:row.update(NFE=50,t_start=50,complete_posterior_schedule=True)
    return runner.save_frame(directory,binding,spec,source,RUNTIME,row,source['rgb'],wave,wave,
                             hifi_receipt=HIFI)


class CostTests(unittest.TestCase):
    def test_registered_grid_and_summary(self):
        rows=[]
        for stage in k.STATUS:
            self.assertEqual(len(k.specs(stage)),72 if stage=='own' else 48)
            for spec in k.specs(stage):
                r=runner.row_base(spec,sample(spec['source_index']),RUNTIME)
                r.update(RX_seconds=spec['source_index']+1,header_accepted=True)
                rows.append(r)
        result=k.summarize(rows)
        self.assertEqual(len(result),30);self.assertTrue(all(r['frames']==4 and r['RX_mean_seconds']==2.5 for r in result))
        with self.assertRaises(RuntimeError):k.summarize(rows[:-1])
        with self.assertRaises(RuntimeError):k.summarize([*rows[:-1],rows[0]])

    def test_true_float_waveform_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            s=sample();spec=k.specs('own')[0];p,v=make_frame(Path(folder),'bound',spec,s)
            k.validate_frame(v,'bound',spec,s,RUNTIME)
            altered=copy.deepcopy(v);altered['row']['reference_sha256']='0'*64
            altered['payload_sha256']=c.identity({kk:vv for kk,vv in altered.items() if kk!='payload_sha256'})
            with self.assertRaises(RuntimeError):k.validate_frame(altered,'bound',spec,s,RUNTIME)
            with open(v['archive'],'ab') as stream:stream.write(b'changed')
            with self.assertRaises(RuntimeError):k.validate_frame(v,'bound',spec,s,RUNTIME)

    def test_hifi_full_sampler_not_two_step_probe(self):
        spec=k.specs('external')[1];s=sample()
        with tempfile.TemporaryDirectory() as folder:
            _,v=make_frame(Path(folder),'bound',spec,s)
            v['hifi_receipt']['diagnostic_probe']=True
            v['payload_sha256']=c.identity({kk:vv for kk,vv in v.items() if kk!='payload_sha256'})
            with self.assertRaises(RuntimeError):k.validate_frame(v,'bound',spec,s,RUNTIME)

    def test_receiver_timing_excludes_tx_and_wraps_both_syncs(self):
        events=[];source=sample();spec=k.specs('own')[0]
        wave=np.ones((1024,2),np.float32)
        class Native:
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:events.append('sync')))
            def prepare(self,s):events.append('TX');return wave,wave.copy(),{}
            def receive(self,y,n,s,m):events.append('RX');return source['rgb'],dict(header_accepted='not_applicable')
        def clock():events.append('clock');return float(len(events))
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(runner.time,'perf_counter',side_effect=clock):
            runner.measured_own(Native(),spec,source,RUNTIME,Path(folder),'bound')
        self.assertEqual(events,['TX','sync','clock','RX','sync','clock'])

    def test_calibration_population_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);sources=[sample(i) for i in range(4)];archive=out/'population.npz'
            digest=c.atomic_npz(archive,source_rgb=np.stack([s['rgb'] for s in sources]))
            reg=dict(protocol=k.PROTOCOL,source_indices=[0,1,2,3],sources=[{kk:vv for kk,vv in s.items() if kk!='rgb'} for s in sources],archive_sha256=digest)
            self.assertEqual(len(k.validate_population(reg,archive)),4)
            reg['sources'][0]['image_id']='different'
            # Source identity is pinned by the outer registration. A target-pixel change is also independently detected.
            reg['sources'][0]['rgb_sha256']='0'*64
            with self.assertRaises(RuntimeError):k.validate_population(reg,archive)

    def test_finished_stage_binds_completion_and_real_warmups(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);sources=[sample(i) for i in range(4)];archive=out/'population.npz'
            digest=c.atomic_npz(archive,source_rgb=np.stack([s['rgb'] for s in sources]))
            pop=dict(protocol=k.PROTOCOL,source_indices=[0,1,2,3],sources=[{kk:vv for kk,vv in s.items() if kk!='rgb'} for s in sources],archive=str(archive),archive_sha256=digest)
            c.seal(out/'population.json',pop);bindings={str(out/'population.json'):c.sha(out/'population.json'),str(archive):digest}
            binding,rp=runner.registration(out,'own',bindings,RUNTIME,pop)
            warmups={};frames=[]
            for spec in k.specs('own'):
                group=(spec['N'],spec['method']);source=sources[spec['source_index']]
                if group not in warmups:
                    ws=dict(spec,source_index=0,snr_db=13)
                    wp,_=make_frame(out/'warmups'/runner.name(ws),binding,ws,sources[0]);warmups[group]=wp
                p,_=make_frame(out/'frames'/runner.name(spec),binding,spec,source)
                c.seal(p.parent/'warmup.json',dict(path=str(warmups[group]),sha256=c.sha(warmups[group])));frames.append(p)
            runner.finish(out,'own',rp,binding,bindings,frames,list(warmups.values()),lambda *a,**kw:None)
            rows,reg,bb=k.load_stage(out,'own')
            self.assertEqual(len(rows),72);self.assertIn(str(out/'own_completion.json'),bb)
            reg['runtime']['uuid']='GPU-other';c.write(rp,reg)
            with self.assertRaises(RuntimeError):k.load_stage(out,'own')

    def test_native_frozen_source_signatures(self):
        # Parse actual historical sources without importing Torch or constructing a model.
        root=Path(__file__).resolve().parents[2]/'historical_eval_20261003/source'
        import os
        if os.environ.get('OWN_CONTROLS_TEST_ROOT'):root=Path(os.environ['OWN_CONTROLS_TEST_ROOT'])
        if not root.exists():self.skipTest('Historical source mirror not present; set OWN_CONTROLS_TEST_ROOT')
        expected={
            'experiments/extreme-bandwidth-20261001-N1024/digital_protocol.py':{'transmit':['scales','label','action'],'receive':['y','snr','phy'],'apply_channel':['wave','snr','source_id','seed','phy']},
            'experiments/extreme-bandwidth-20261001-N1024/train.py':{'load_for_evaluation':['selected_path','scale','device']},
            'experiments/token_channel_efficiency_20260923/src/token_efficiency/execution.py':{'receive':['observed','snr','cell','vae','var','decoder','device','model'],'load_selected_budget':['selected_path','scale','device']},
            'experiments/scale-causal-partial-residual-20261002/partial_receiver.py':{'complete_partial':['vae','var','event','device','return_logits']},
            'experiments/scale-causal-partial-residual-20261002/m1_runner.py':{'waveform':['loaded','scales','a','memo']}}
        for file,functions in expected.items():
            tree=ast.parse((root/file).read_text(encoding='utf-8'))
            found={node.name:[a.arg for a in node.args.args] for node in tree.body if isinstance(node,ast.FunctionDef)}
            for name,args in functions.items():self.assertEqual(found[name],args,(file,name))

    def test_external_stage_requires_native_full_pair_timing_proof(self):
        from external_eval import base_row
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);sources=[sample(i) for i in range(4)];archive=out/'population.npz'
            digest=c.atomic_npz(archive,source_rgb=np.stack([s['rgb'] for s in sources]))
            pop=dict(protocol=k.PROTOCOL,source_indices=[0,1,2,3],sources=[{kk:vv for kk,vv in s.items() if kk!='rgb'} for s in sources],archive=str(archive),archive_sha256=digest)
            c.seal(out/'population.json',pop);bindings={str(out/'population.json'):c.sha(out/'population.json'),str(archive):digest}
            binding,rp=runner.registration(out,'external',bindings,RUNTIME,pop)
            warmups={};frames=[]
            def pair(base,directory):
                source=sources[base['source_index']];wave=np.ones((base['N'],2),np.float32);images=np.stack([source['rgb']]*2)
                npz=directory/'native.npz';digest=c.atomic_npz(npz,images=images,observed=wave,transmitted=wave)
                rr=[]
                for j,m in enumerate(c.METHODS):
                    row=base_row(base,m,source,c.rgb_sha(source['rgb']))
                    row.update(image_sha256=c.rgb_sha(images[j]),observed_sha256=c.array_sha(wave),transmitted_sha256=c.array_sha(wave),
                        header_accepted=True,audit_used_to_control_receiver=False,TX_seconds=.2,RX_seconds=.1,
                        NFE=50 if j else 0,t_start=50 if j else '',complete_posterior_schedule=bool(j))
                    rr.append(row)
                raw=dict(binding=binding,frame=base,source_id=source['image_id'],reference_sha256=c.rgb_sha(source['rgb']),
                    rows=rr,archive=str(npz),archive_sha256=digest,observed_sha256=c.array_sha(wave),transmitted_sha256=c.array_sha(wave),hifi_receipt=HIFI)
                raw['payload_sha256']=c.identity(raw);c.validate_frame_receipt(raw,binding,base,source)
                rp=directory/'native.json';c.seal(rp,raw);bindings[str(rp)]=c.sha(rp);bindings[str(npz)]=digest
                result={}
                for j,m in enumerate(c.METHODS):
                    spec=dict(base,method=m);row=runner.row_base(spec,source,RUNTIME)
                    row.update({kk:rr[j][kk] for kk in ('RX_seconds','header_accepted','NFE','complete_posterior_schedule')});row['t_start']=50 if j else 0
                    path,_=runner.save_frame(directory/m,binding,spec,source,RUNTIME,row,images[j],wave,wave,
                        native_receipt=str(rp),native_receipt_sha256=c.sha(rp),hifi_receipt=HIFI)
                    result[m]=path
                return result
            for n in (1024,2048):warmups[n]=pair(dict(source_index=0,N=n,snr_db=13,noise_seed=4101),out/'warmups'/str(n))
            for i in range(4):
                for n in (1024,2048):
                    for s in (1,7,13):
                        paths=pair(dict(source_index=i,N=n,snr_db=s,noise_seed=4101),out/'frames'/f'{i}-{n}-{s}')
                        for m,p in paths.items():c.seal(p.parent/'warmup.json',dict(path=str(warmups[n][m]),sha256=c.sha(warmups[n][m])))
                        frames+=list(paths.values())
            runner.finish(out,'external',rp,binding,bindings,frames,[p for group in warmups.values() for p in group.values()],lambda *a,**kw:None)
            rr,reg,bb=k.load_stage(out,'external');self.assertEqual(len(rr),48)
            # Re-sealing a changed cost row cannot override the independent original timed pair.
            changed=c.read(frames[0]);changed['row']['RX_seconds']=999
            changed['payload_sha256']=c.identity({kk:vv for kk,vv in changed.items() if kk!='payload_sha256'});c.write(frames[0],changed)
            done=c.read(out/'external_completion.json');done['outputs'][str(frames[0])]=c.sha(frames[0]);c.write(out/'external_completion.json',done)
            with self.assertRaises(RuntimeError):k.load_stage(out,'external')


if __name__=='__main__':unittest.main()
