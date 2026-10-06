"""Synthetic population/codec/receipt tests: no development data or GPU."""
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import development_source_common as c
import development100_assets as a
import development100_source_driver as s

HERE=Path(__file__).resolve().parent
FULL=HERE if (HERE/'full1000_source_driver.py').is_file() else HERE.parent/'full_calibration'
PHY=HERE if (HERE/'h64_source.py').is_file() else HERE.parent/'phy_codec'
OLD_SOURCE=HERE/'h_source_driver.py' if (HERE/'h_source_driver.py').is_file() else HERE.parent/'h_source_driver.py'
for p in (FULL,PHY):sys.path.insert(0,str(p))
f=c.module(FULL/'full1000_source_driver.py','dev_test_frozen_source')
av=c.module(FULL/'full1000_assets.py','dev_test_frozen_assets')


def fixture():
    pixels=np.arange(3*256*256,dtype=np.uint8).reshape(3,256,256)
    pre=hashlib.sha256(pixels.tobytes()).hexdigest();ids=[f'dev{i:03d}' for i in range(100)]
    pop=dict(stage='m1_development',calibration_or_development='m1_development',identity={'frozen':'same'},source_ids=ids,
        preprocessing_ids=[pre]*100,data_bindings=[dict(index=i,rgb_sha256=pre,source_npz_sha256='a'*64) for i in range(100)])
    cal=dict(stage='m1_calibration',calibration_or_development='m1_calibration',identity=pop['identity'],source_ids=[f'cal{i}' for i in range(1000)])
    targets=[dict(image_id=sid,class_index=i) for i,sid in enumerate(ids)]
    return pop,cal,targets,pixels


class SourceTests(unittest.TestCase):
    def test_population_list_order_disjointness_and_roles(self):
        p,cal,t,_=fixture();rows=a.build_plan(p,cal,t,p['source_ids'],['target_development']*100,list(range(100)),'/root/outputs/VAR-PROGRESSIVE-CHANNEL-001')
        self.assertEqual(len(rows),100);self.assertEqual(Path(rows[99]['original_archive']).parts[-3:],('images','099','reconstructions.npz'))
        for bad in ('duplicate','overlap','order','dict','role','visual'):
            pp=copy.deepcopy(p);cc=copy.deepcopy(cal);roles=['target_development']*100
            if bad=='duplicate':pp['source_ids'][1]=pp['source_ids'][0]
            if bad=='overlap':cc['source_ids'][0]=p['source_ids'][0]
            if bad=='order':pp['data_bindings'][0]['index']=1
            if bad=='dict':pp['data_bindings']={'latents':{}}
            if bad=='role':roles[5]='calibration'
            if bad=='visual':cc['identity']={}
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):
                a.build_plan(pp,cc,t,p['source_ids'],roles,list(range(100)),'/root/outputs/VAR-PROGRESSIVE-CHANNEL-001')

    def test_rgb_original_conversion_and_pixel_hash_are_exact(self):
        p,_,_,pixels=fixture();original=pixels.astype(np.float32)/255
        self.assertTrue(np.array_equal(a.pixels_from_original(original,p['preprocessing_ids'][0]),pixels))
        for bad in (original.transpose(1,2,0),np.full(original.shape,np.nan,np.float32),original-1):
            with self.assertRaises(RuntimeError):a.pixels_from_original(bad,p['preprocessing_ids'][0])
        with self.assertRaisesRegex(RuntimeError,'preprocessing'):
            a.pixels_from_original(np.zeros_like(original),p['preprocessing_ids'][0])

    def test_only_dev100_bytes_read_from_a_mixed1200_container(self):
        values=np.arange(1200*680,dtype=np.int64).reshape(1200,680);stream=io.BytesIO();np.save(stream,values,allow_pickle=False)
        raw=stream.getvalue();probe=io.BytesIO(raw);version=np.lib.format.read_magic(probe)
        np.lib.format.read_array_header_1_0(probe);limit=probe.tell()+100*680*8
        class Guard(io.BytesIO):
            def read(self,n=-1):
                if n<0 or self.tell()+n>limit:raise AssertionError('Non-development token bytes requested')
                return super().read(n)
        guard=Guard(raw);got,proof=a.read_npy_prefix(guard)
        self.assertTrue(np.array_equal(got,values[:100]));self.assertEqual(guard.tell(),limit)
        self.assertEqual(proof['other_rows_read'],0)
        with self.assertRaises(RuntimeError):a.read_npy_prefix(io.BytesIO(raw),101)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'mixed.npz';np.savez_compressed(p,tokens=values)
            read,_=a.cache_prefix(p,'tokens');self.assertTrue(np.array_equal(read,values[:100]))

    def test_asset_writer_and_codec_writer_feed_frozen_downstream_reader(self):
        p,cal,t,pixels=fixture();tokens=np.arange(680,dtype=np.int64)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);areg=root/'assets_reg.json';sreg=root/'source_reg.json';c.save(areg,{});c.save(sreg,{})
            row=a.build_plan(p,cal,t,p['source_ids'],['target_development']*100,list(range(100)),root)[0]
            record,ao=a.write_asset(root/'assets',c.sha(areg),row,tokens,pixels,av)
            ctx=dict(cfg=dict(out=str(root/'source'),registration=str(sreg),assets_registration=str(areg)),records=[record],
                pop=p,assets_done={'outputs':ao},assets_api=av,codec_runner=f,ids=p['source_ids'])
            cp,flat,px=s.load_asset(ctx,0);self.assertTrue(np.array_equal(flat,tokens));self.assertTrue(np.array_equal(px,pixels))
            bits={f'm{m}_bits':np.array([0,1,0,1],np.uint8) for m in f.MODES}
            lengths=[dict(m=m,raw_bits=12*sum(x*x for x in f.SIZES[:m]),arithmetic_bits=4,flush_bits=2,zero_extension_reads=30) for m in f.MODES]
            encoded,co=s.write_source(ctx,0,cp,bits,lengths)
            downstream=c.module(FULL/'h_full_payload_driver.py','dev_downstream_source_contract')
            core=SimpleNamespace(initial=SimpleNamespace(split_raw_tokens=lambda x:[x.copy()]),phy=SimpleNamespace(binary=np.asarray),
                token_count=lambda m:sum(x*x for x in f.SIZES[:m]))
            result=downstream.load_source(dict(cfg={},source=dict(records=[encoded],registration_sha256=c.sha(sreg),outputs=co),
                manifest={'records':[record]},context={'source_ids':p['source_ids']},core=core),0)
            self.assertTrue(np.array_equal(result['scales'][0],tokens))
            self.assertEqual(set(result['arithmetic_bits']),{6,7,8,9})
            # A changed source identity remains invalid even if its JSON SHA is updated.
            cp['source_id']='cal0';Path(record['checkpoint']).write_text(__import__('json').dumps(cp))
            record['checkpoint_sha256']=c.sha(record['checkpoint']);ao[record['checkpoint']]=record['checkpoint_sha256']
            with self.assertRaisesRegex(RuntimeError,'mapping'):s.load_asset(ctx,0)

    def test_original_codec_uses_one_TX_four_independent_RX_contexts(self):
        from test_h64_core import load_primitives,Provider
        import h64_source
        source=c.module(OLD_SOURCE,'dev_source_original_provider')
        calls=[];arrays,rows=f.encode_new(np.arange(680,dtype=np.int64),source,h64_source,lambda:Provider(calls,True),load_primitives())
        self.assertEqual([len(x.history) for x in calls],[9,6,7,8,9]);self.assertEqual(len(arrays),4)
        self.assertTrue(all(r['zero_extension_reads']==30 for r in rows))

    def test_encoder_exact_formula_and_mismatch_must_stop(self):
        # Small tensor facade verifies the arithmetic/input order without torch/GPU.
        class Tensor:
            def __init__(self,x):self.x=np.asarray(x)
            def __truediv__(self,v):return Tensor(self.x/v)
            def __sub__(self,v):return Tensor(self.x-v)
            def __getitem__(self,k):return Tensor(self.x[k])
            def cpu(self):return self
            def numpy(self):return self.x
        _,_,_,pixels=fixture();expected=np.arange(680,dtype=np.int64);seen=[]
        def encoder(x):seen.append(x.x.copy());return x
        quant=SimpleNamespace(f_to_idxBl_or_fhat=lambda x,to_fhat:[Tensor(expected[None])])
        torch=SimpleNamespace(float32=np.float32,as_tensor=lambda x,dtype,device:Tensor(np.asarray(x,dtype)),
            cat=lambda xs,dim:Tensor(np.concatenate([x.x for x in xs],axis=dim)))
        native=SimpleNamespace(torch=torch,loaded=dict(device='fixture',vae=SimpleNamespace(encoder=encoder,quant_conv=lambda x:x,quantize=quant)))
        self.assertTrue(np.array_equal(s.verify_encoder_tokens(native,pixels,expected),expected))
        self.assertTrue(np.array_equal(seen[0],pixels[None].astype(np.float32)/127.5-1))
        bad=expected.copy();bad[0]=2
        with self.assertRaisesRegex(RuntimeError,'preserve and stop'):s.verify_encoder_tokens(native,pixels,bad)

    def test_complete_codec_cannot_claim_partial_population_complete(self):
        ids=[f'd{i}' for i in range(100)];records=[dict(source_index=i,source_id=sid) for i,sid in enumerate(ids)]
        result=s.finish(records,ids);self.assertEqual(result['new_canonical_roundtrips'],400)
        self.assertEqual(result['new_packet_decodes'],0);self.assertFalse(result['policy_selection'])
        with self.assertRaises(RuntimeError):s.finish(records[:-1],ids)
        with self.assertRaises(RuntimeError):s.finish(records,list(reversed(ids)))

    def test_empty_owner_created_directory_once_only(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out';out.mkdir();c.claim(out,'a'*64,'assets');before=(out/'attempt.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError,'Prior'):c.claim(out,'a'*64,'assets')
            self.assertEqual((out/'attempt.json').read_bytes(),before)

    def test_closed_or_unfinished_packet_activity_cannot_be_hidden_by_source_stage(self):
        b=dict(created=True,unresolved=0,failed=0,phase_charged={'development':0},development_remaining=13200,charged=153960)
        c.budget_unchanged(b,{'budget_before':copy.deepcopy(b)})
        for key,value in (('charged',153961),('unresolved',1),('failed',1),('development_remaining',13199)):
            with self.subTest(key=key),self.assertRaises(RuntimeError):c.budget_unchanged(dict(b,**{key:value}),{'budget_before':b})

    def test_frozen_native_flags_and_source_graph_are_reused_unmodified(self):
        ctx=dict(cal={'identity':{'models':'frozen'}},expected_flags={'threads':6},static_bindings={'old.py':'a'},bound={'old.py':'a'})
        native=SimpleNamespace(loaded={'identity':{'models':'frozen'}},flags={'threads':6},driver_bindings={'old.py':'a'})
        f.validate_native(native,ctx);native.driver_bindings['other.py']='b'
        with self.assertRaisesRegex(RuntimeError,'source closure'):f.validate_native(native,ctx)

    def test_unregistered_input_stops_before_cache_or_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cfg.json';r=Path(td)/'reg.json';c.save(r,{'status':'NOT_REGISTERED'})
            c.save(p,dict(schema='H_DEVELOPMENT100_PREPARATION_CONFIG_V1',kind='assets',registration=str(r)))
            with self.assertRaisesRegex(RuntimeError,'registration'):a.load_registered(p)

    def test_actual_exclusive_GPU_owner_identity_config_and_admission_required(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reg=root/'reg.json';config=root/'config.json';owner_config=root/'owner.json'
            c.save(reg,{});c.save(config,{})
            argv=['python','-B',str(Path(s.__file__).resolve()),'--config',str(config.resolve())]
            parent=dict(pid=100,start_ticks=10,uid=1002,argv=['owner']);child=dict(pid=101,start_ticks=11,uid=1002,argv=argv)
            out=root/'source';base=root/'owner';job=dict(id='source100',argv=argv,out=str(out),completion=str(out/'completion.json'))
            oc=dict(registration=str(reg),out=str(root),owner_out=str(base),stages=[dict(id='source',resource='gpu',jobs=[job])],
                gpu_threads=6,gpu_affinity=[24,25,26,27,28,29],gpu_device=0)
            c.save(owner_config,oc);ip=base/'owner_identity.json'
            c.save(ip,dict(parent,registration_sha256=c.sha(reg),config_sha256=c.sha(owner_config)))
            lp=base/'stages/source/workers/source100/launch.json'
            c.save(lp,dict(identity=child,registration_sha256=c.sha(reg),argv=argv,resource='gpu',threads=6,affinity=oc['gpu_affinity']))
            gp=base/'stages/source/gpu_admission.json';c.save(gp,dict(status='GPU_IDLE_CONFIRMED',registration_sha256=c.sha(reg)))
            api=SimpleNamespace(validate_config=lambda *_:None,command_entry=lambda v:Path(v[2]),
                same_identity=lambda x,y:all(x[k]==y[k] for k in ('pid','start_ticks','uid','argv')),
                identity=lambda pid:parent if pid==100 else child)
            ctx=dict(cfg=dict(owner_config=str(owner_config),registration=str(reg),H_out=str(root),out=str(out)),reg={},owner=api)
            with patch.object(c.os,'getppid',return_value=100),patch.object(c.os,'getpid',return_value=101),\
                patch.object(c.os,'sched_getaffinity',return_value=set(oc['gpu_affinity']),create=True),patch.dict(c.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                got=c.live_owner(ctx,config,s.__file__,'source');self.assertEqual(got['worker_identity'],child)
                gp.write_text('{"status":"NOT_IDLE"}')
                with self.assertRaisesRegex(RuntimeError,'admission'):c.live_owner(ctx,config,s.__file__,'source')
                gp.write_text(__import__('json').dumps(dict(status='GPU_IDLE_CONFIRMED',registration_sha256=c.sha(reg))))
                bad=c.read(ip);bad['config_sha256']='f'*64;ip.write_text(__import__('json').dumps(bad))
                with self.assertRaisesRegex(RuntimeError,'parent'):c.live_owner(ctx,config,s.__file__,'source')

    def test_source_boundary_STOP_preserves_without_calling_any_codec(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);stop=root/'STOP';stop.write_text('stop')
            cfg=dict(out=str(root/'source'),stop_file=str(stop),max_seconds=60)
            with self.assertRaisesRegex(RuntimeError,'STOP'):
                c.source_boundary({'cfg':cfg},{},0,0)

if __name__=='__main__':unittest.main()
