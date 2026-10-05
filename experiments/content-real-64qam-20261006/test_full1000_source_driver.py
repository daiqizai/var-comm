"""CPU tests only: frozen arithmetic primitives, synthetic CDFs/assets/identities."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
PHY_DIR=HERE if (HERE/'h64_source.py').is_file() else HERE.parent/'phy_codec'
OLD_SOURCE=HERE/'h_source_driver.py' if (HERE/'h_source_driver.py').is_file() else HERE.parent/'h_source_driver.py'
sys.path.insert(0,str(PHY_DIR))
spec=importlib.util.spec_from_file_location('full1000_source_tested',HERE/'full1000_source_driver.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
spec=importlib.util.spec_from_file_location('full1000_assets_test_reference',HERE/'full1000_assets.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class SourceTests(unittest.TestCase):
    def test_source_only_registration_rejects_later_closed_packet_activity(self):
        before=dict(created=True,charged=69960,phase_charged={'initial_true200':19200},unresolved=0,failed=0,development_remaining=13200)
        reg={'budget_before':copy.deepcopy(before)};d.verify_frozen_budget(before,reg)
        changed=copy.deepcopy(before);changed['charged']+=1;changed['phase_charged']['initial_true200']+=1
        with self.assertRaisesRegex(RuntimeError,'changed since'):d.verify_frozen_budget(changed,reg)
        with self.assertRaisesRegex(RuntimeError,'changed since'):d.verify_frozen_budget(dict(before,unresolved=1),reg)
        with self.assertRaisesRegex(RuntimeError,'changed since'):d.verify_frozen_budget(before,{})

    def test_original_canonical_core_independent_contexts_and_all_prefixes(self):
        from test_h64_core import load_primitives,Provider
        import h64_source as codec
        source=d.module(OLD_SOURCE,'frozen_source_for_cpu_test')
        flat=np.arange(680,dtype=np.int64);providers=[]
        arrays,rows=d.encode_new(flat,source,codec,lambda:Provider(providers,True),load_primitives())
        self.assertEqual(len(providers),5)
        self.assertEqual([len(p.history) for p in providers],[9,6,7,8,9])
        self.assertEqual(set(arrays),{'m6_bits','m7_bits','m8_bits','m9_bits'})
        self.assertTrue(all(r['zero_extension_reads']==30 for r in rows))
        # The injected frozen source adapter's renderer is never consulted.
        source.render_received=lambda *_:(_ for _ in ()).throw(AssertionError('Image rendering forbidden'))
        arrays2,rows2=d.encode_new(flat,source,codec,lambda:Provider([],True),load_primitives())
        self.assertEqual(rows,rows2)
        for key in arrays:self.assertTrue(np.array_equal(arrays[key],arrays2[key]))

    def test_bad_canonical_or_wrong_source_roundtrip_stops_without_substitution(self):
        flat=np.arange(680,dtype=np.int64)
        sizes=d.SIZES;ends=np.cumsum([0]+[s*s for s in sizes])
        source=SimpleNamespace(split_tokens=lambda x:[x[ends[i]:ends[i+1]].copy() for i in range(10)])
        def encode(scales,*_):
            return {m:dict(bits=np.array([1,0],np.uint8),arithmetic_bits=2,raw_bits=12*int(ends[m]),flush_bits=2) for m in d.MODES}
        def decode(bits,m,*_):
            return dict(scales=source.split_tokens(flat)[:m],canonical=True,status='SOURCE_DECODED',payload_bits=2,zero_extension_reads=30)
        codec=SimpleNamespace(encode_prefixes=encode,decode_prefix=decode)
        d.encode_new(flat,source,codec,lambda:None,None)
        codec.decode_prefix=lambda *args:dict(decode(*args),canonical=False)
        with self.assertRaisesRegex(RuntimeError,'canonical'):d.encode_new(flat,source,codec,lambda:None,None)
        def wrong(*args):
            result=decode(*args);result['scales'][0][0]^=1;return result
        codec.decode_prefix=wrong
        with self.assertRaisesRegex(RuntimeError,'roundtrip differs'):d.encode_new(flat,source,codec,lambda:None,None)
        codec.decode_prefix=lambda *_:(_ for _ in ()).throw(RuntimeError('CDF model failed'))
        with self.assertRaisesRegex(RuntimeError,'CDF model failed'):d.encode_new(flat,source,codec,lambda:None,None)

    def test_reuse_requires_sealed_source_identity_bits_and_canonical_length(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);out=base/'assets';archive=out/'sources/0005.npz';codec=out/'codec_reuse/0005.npz'
            tokens=np.arange(680,dtype=np.int64);pixels=np.zeros((3,256,256),np.uint8);pre=d.hashlib.sha256(pixels.tobytes()).hexdigest()
            a.save_npz(archive,tokens=tokens,pixels=pixels)
            bits={f'm{m}_bits':np.array([0,1,1,0],np.uint8) for m in d.MODES};a.save_npz(codec,**bits)
            lengths=[dict(m=m,raw_bits=12*sum(s*s for s in d.SIZES[:m]),arithmetic_bits=4,flush_bits=2,zero_extension_reads=30) for m in d.MODES]
            old=base/'original.json';d.save(old,{'original':'not regenerated'})
            registration=base/'reg.json';d.save(registration,{'registered':True})
            cp_path=out/'source_checkpoints/0005.json'
            cp=dict(status='H_FULL1000_SOURCE_ASSET_READY',registration_sha256=d.sha(registration),source_index=5,source_id='source5',
                preprocessing_id=pre,tokens_sha256=a.token_sha(tokens),archive=str(archive),codec_archive=str(codec),
                legacy_source200_index=99,codec_status='SOURCE_CODEC_REUSED_BY_EXACT_ID',original_calibration_index=5,new_source_encoding=False,
                outputs={str(archive):d.sha(archive),str(codec):d.sha(codec)},
                reuse_proof=dict(status='SOURCE_CODEC_REUSED_BY_EXACT_ID',source_id='source5',full_source_index=5,legacy_source200_index=99,
                    preprocessing_id=pre,tokens_sha256=a.token_sha(tokens),independent_roundtrip_reused=True,new_canonical_decode=False,
                    clean_image_keys_read=False,original_input_bindings={str(old):d.sha(old)},original_lengths=lengths))
            d.save(cp_path,cp)
            record={k:cp[k] for k in ('source_index','source_id','preprocessing_id','tokens_sha256','archive','codec_status','codec_archive','legacy_source200_index')}
            record.update(checkpoint=str(cp_path),checkpoint_sha256=d.sha(cp_path))
            ctx=dict(records=[None]*5+[record],assets_done=dict(outputs=dict(cp['outputs'],**{str(cp_path):d.sha(cp_path)}),input_bindings={str(old):d.sha(old)}),
                     assets_api=a,cfg=dict(assets_registration=str(registration)))
            result,flat,cached=d.load_source(ctx,5)
            self.assertEqual(result['legacy_source200_index'],99);self.assertTrue(np.array_equal(flat,tokens))
            for key in bits:self.assertTrue(np.array_equal(cached[0][key],bits[key]))
            # Real prepared downstream reader consumes the exact writer result.
            downstream=d.module(HERE/'h_full_payload_driver.py','full1000_downstream_cpu_interface')
            emitted,outputs=d.write_source(base/'source',d.sha(registration),result,cp_path,*cached,'EXACT_ID_SOURCE200_REUSE',a)
            core=SimpleNamespace(initial=SimpleNamespace(split_raw_tokens=lambda x:[x.copy()]),
                phy=SimpleNamespace(binary=lambda x:np.asarray(x,dtype=np.uint8)),token_count=lambda m:sum(s*s for s in d.SIZES[:m]))
            downstream_ctx=dict(cfg={},source=dict(records=[None]*5+[emitted],registration_sha256=d.sha(registration),outputs=outputs),
                manifest={'records':[None]*5+[record]},context={'source_ids':[None]*5+['source5']},core=core)
            loaded=downstream.load_source(downstream_ctx,5)
            self.assertTrue(np.array_equal(loaded['scales'][0],tokens))
            for m in d.MODES:self.assertTrue(np.array_equal(loaded['arithmetic_bits'][m],bits[f'm{m}_bits']))
            broken=copy.deepcopy(cp);broken['reuse_proof']['legacy_source200_index']=5
            cp_path.write_text(d.json.dumps(broken));record['checkpoint_sha256']=d.sha(cp_path);ctx['assets_done']['outputs'][str(cp_path)]=d.sha(cp_path)
            with self.assertRaisesRegex(RuntimeError,'reuse proof'):d.load_source(ctx,5)
            with self.assertRaisesRegex(RuntimeError,'canonical evidence'):d.validate_lengths(bits,[dict(r,zero_extension_reads=29) for r in lengths])

    def test_population_completion_cannot_upgrade_source200_to1000(self):
        ids=[f'id{i}' for i in range(1000)]
        records=[dict(source_index=i,source_id=sid,codec_origin='EXACT_ID_SOURCE200_REUSE' if i%5==0 else 'NEW_FROZEN_SOURCE_ENCODING') for i,sid in enumerate(ids)]
        counts=d.finish_records(records,ids);self.assertEqual(counts['newly_encoded_sources'],800)
        with self.assertRaisesRegex(RuntimeError,'coverage'):d.finish_records(records[:200],ids)
        bad=copy.deepcopy(records);bad[1]['codec_origin']='EXACT_ID_SOURCE200_REUSE'
        with self.assertRaisesRegex(RuntimeError,'counts'):d.finish_records(bad,ids)
        bad=copy.deepcopy(records);bad[1]['source_id']=ids[0]
        with self.assertRaisesRegex(RuntimeError,'order'):d.finish_records(bad,ids)

    def test_owner_precreated_directory_once_only_and_old_outputs_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'source';out.mkdir();d.claim_source_output(out,'a'*64)
            original=(out/'attempt.json').read_bytes()
            self.assertEqual(d.read(out/'attempt.json')['status'],'H_FULL1000_SOURCE_CODEC_ATTEMPT')
            with self.assertRaisesRegex(RuntimeError,'independent recovery'):d.claim_source_output(out,'a'*64)
            self.assertEqual((out/'attempt.json').read_bytes(),original)

    def test_visual_identity_numeric_flags_and_complete_closure_all_match(self):
        ctx=dict(cal={'identity':{'VAR':'frozen'}},expected_flags={'threads':6,'interop_threads':2},
                 static_bindings={'old.py':'a','newly_tracked.py':'b'},bound={'old.py':'a','newly_tracked.py':'b'})
        native=SimpleNamespace(loaded={'identity':{'VAR':'frozen'}},flags=ctx['expected_flags'].copy(),driver_bindings=ctx['static_bindings'].copy())
        d.validate_native(native,ctx);native.flags['threads']=2
        with self.assertRaisesRegex(RuntimeError,'numeric flags'):d.validate_native(native,ctx)
        native.flags=ctx['expected_flags'].copy();ctx['bound'].pop('newly_tracked.py')
        with self.assertRaisesRegex(RuntimeError,'complete visual source closure'):d.validate_native(native,ctx)


class OwnerTests(unittest.TestCase):
    def fixture(self,base):
        reg=base/'reg.json';cfg=base/'driver.json';d.save(reg,{'registered':True});d.save(cfg,{'prepared':True})
        parent=dict(pid=100,start_ticks=10,uid=1002,argv=['python','owner.py'])
        argv=['python','-B',str(Path(d.__file__).resolve()),'--config',str(cfg)]
        child=dict(pid=101,start_ticks=11,uid=1002,argv=argv);out=base/'source';owner_out=base/'owner'
        job=dict(id='source1000',argv=argv,out=str(out),completion=str(out/'completion.json'))
        ownercfg=dict(registration=str(reg),owner_out=str(owner_out),out=str(base),stages=[dict(id='source',resource='gpu',jobs=[job])],
                      gpu_threads=6,gpu_affinity=list(range(24,30)),gpu_device=0)
        oc=base/'owner_config.json';d.save(oc,ownercfg)
        d.save(owner_out/'owner_identity.json',dict(parent,registration_sha256=d.sha(reg),config_sha256=d.sha(oc)))
        launch=owner_out/'stages/source/workers/source1000/launch.json'
        d.save(launch,dict(identity=child,registration_sha256=d.sha(reg),resource='gpu',argv=argv,threads=6,affinity=ownercfg['gpu_affinity']))
        gp=owner_out/'stages/source/gpu_admission.json';d.save(gp,dict(status='GPU_IDLE_CONFIRMED',registration_sha256=d.sha(reg)))
        api=SimpleNamespace(validate_config=lambda *_:None,command_entry=lambda args:Path(args[2]).resolve(),
            identity=lambda pid:copy.deepcopy(parent if pid==100 else child),same_identity=lambda a,b:all(a[k]==b[k] for k in ('pid','start_ticks','uid','argv')))
        return dict(cfg=dict(registration=str(reg),source_owner_config=str(oc),out=str(out),H_out=str(base)),reg={},owner=api),cfg,launch,gp
    def call(self,ctx,cfg):
        with patch.object(d.os,'getppid',return_value=100),patch.object(d.os,'getpid',return_value=101),\
             patch.object(d.os,'sched_getaffinity',return_value=set(range(24,30)),create=True),patch.dict(d.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            return d.verify_live_visual_owner(ctx,cfg)
    def test_requires_current_exact_owner_child_identity_and_gpu_admission(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,cfg,launch,gp=self.fixture(Path(td));result=self.call(ctx,cfg)
            self.assertEqual(result['worker_identity']['pid'],101)
            changed=d.read(launch);changed['identity']['start_ticks']+=1;launch.write_text(d.json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError,'launch/runtime'):self.call(ctx,cfg)
    def test_foreign_parent_or_busy_admission_refused(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,cfg,launch,gp=self.fixture(Path(td));original=ctx['owner'].identity
            ctx['owner'].identity=lambda pid:dict(original(pid),uid=3000)
            with self.assertRaisesRegex(RuntimeError,'live parent'):self.call(ctx,cfg)
            ctx['owner'].identity=original;gp.write_text(d.json.dumps(dict(status='GPU_BUSY',registration_sha256=d.sha(ctx['cfg']['registration']))))
            with self.assertRaisesRegex(RuntimeError,'admission absent'):self.call(ctx,cfg)


if __name__=='__main__':unittest.main()
