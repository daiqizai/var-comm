"""Pure metadata and fake-model tests. No Torch, actual image or model loading."""
import copy
from contextlib import nullcontext
import hashlib
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import numpy as np
import p600_received_replay_core as c


def fixture():
    pixels=np.zeros((3,256,256),np.uint8); target=pixels.astype(np.float32)/np.float32(255)
    image=np.full((3,256,256),.25,np.float32);z=np.zeros((32,16,16),np.float32)
    ids=['source'+str(i) for i in range(100)];pre=[c.raw_sha(pixels)]*100
    selected={};scalar={};channels=[]
    for j,(i,s,n) in enumerate(itertools.product(range(100),(1,4,7,13,19),c.SEEDS)):
        row=dict(source_index=i,source_id=ids[i],snr_db=s,noise_seed=n,N=1024,E=2048.,
            preprocessing_id=pre[i],waveform_sha256='a'*64,observation_sha256='b'*64)
        channels.append(row)
        if s in c.SNRS:
            selected[i,s,n]=dict(row,original_tensor_index=j,Z_sha256=c.raw_sha(z),
                existing_image_sha256=c.rgb_sha(image),existing_reference_sha256=c.rgb_sha(target))
            scalar[i,s,n]=dict(row,method='P1024',model_id='P1024_step40000_12b979260ebf',
                base_model_id='P1024_step40000_12b979260ebf',decoder_id='Dc',replay_decoder_id='Dc',label_conditioned='False',
                image_sha256=c.rgb_sha(image),reference_sha256=c.rgb_sha(target),true_class_index='1',
                psnr_db='12.041199826559248',lpips_alex='.12')
    ni=dict(models=dict(decoder=c.DC_STATE));nr=dict(c.FLAGS,torch='fake',cuda='fake')
    bound={'/root/selected_P1024.json':'c'*64};source={'/root/collector.py':'d'*64}
    inv=dict(status='P600_RECEIVED_LATENT_METADATA_VERIFIED',input_bindings=bound,source_bindings=source,
        Z_shape=[1500,32,16,16],Z_dtype='float32',source_indices_shape=[1500],source_indices_dtype='int64',
        original_grid_frames=1500,selected_frames=600,selected_snrs=list(c.SNRS),noise_seeds=list(c.SEEDS),
        Dc_state_sha256=c.DC_STATE,identity=dict(source_ids=ids,preprocessing_ids=pre,role='development',seeds=list(c.SEEDS)),
        selected_rows=list(selected.values()),channel_rows=channels)
    done=dict(status='P600_RECEIVED_LATENT_AUDIT_COMPLETE',source_count=100,frame_count=600,
        input_bindings=bound,source_bindings=source,outputs={'/root/inventory.json':'e'*64})
    for x in (inv,done):
        x.update({k:False for k in ('GPU_used','models_constructed','tensor_values_exported','RGB_parity_verified','ConvNeXt_scored','full_P_reuse_complete')})
        x.update(new_noise_draws=0,new_packet_decodes=0,budget_writes=0)
    metrics=dict(numerical_runtime=c.FLAGS,source_ids=ids,original_replay_inventory=dict(source_ids=ids,
        preprocessing_ids=pre,model_identity=ni))
    nq=dict(status='REAL_NATIVE_QUALIFICATION_PASS',frozen_identity=ni,numerical_runtime=nr)
    docs=(done,inv,dict(rows=list(scalar.values())),metrics,nq)
    ctx=c.context(*docs,inventory_sha256='e'*64)
    return ctx,pixels,image,z,docs


class FakeClassifier:
    def __init__(self):
        self.identity=dict(weights_sha256=c.CONVNEXT,classification_batch_size=1,used_for_selection=False,
            population='development',inference_precision='float32',policy_sha256='c'*64)
        self.calls=[]
    def predict(self,image):self.calls.append(('source',c.rgb_sha(image)));return 1
    def score(self,image,*,true_label,source_prediction):
        self.calls.append(('reconstruction',c.rgb_sha(image)))
        return dict(convnext_true_class=true_label,convnext_source_prediction=source_prediction,
            convnext_prediction=2,convnext_top1_label=False,convnext_source_prediction_agreement=0,
            convnext_used_for_selection=False,convnext_weight_sha256=c.CONVNEXT)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixture()
    def setUp(self):
        self.ctx,self.pixels,self.image,self.z,self.docs=copy.deepcopy(self.base)
        self.classifier=FakeClassifier();self.calls=[]
        self.latents=SimpleNamespace(batch=lambda i,s:np.stack([self.z]*3))
    def decode(self,z):
        self.calls.append(z.copy());return np.stack([self.image]*3)
    def run_source(self):
        return c.replay_source(self.ctx,0,self.pixels,latents=self.latents,decoder=self.decode,classifier=self.classifier)

    def test_original_B3_all6parity_precedes_B1_classification_and_scalar_preserved(self):
        out=self.run_source()
        self.assertEqual(len(self.calls),2);self.assertEqual([x.shape for x in self.calls],[(3,32,16,16)]*2)
        self.assertEqual(len(self.classifier.calls),7)
        self.assertEqual([r['noise_seed'] for r in out['rows']],[2001,2002,2003]*2)
        for row in out['rows']:
            old=self.ctx['scalar'][c.row_key(row)]
            self.assertTrue(all(row[k]==v for k,v in old.items()))
            self.assertIs(row['p_replay_exact_RGB'],True)
        self.assertEqual(out['new_noise_draws'],0);self.assertEqual(out['new_packet_decodes'],0)

    def test_differentRGB_onsecondSNR_stops_before_any_classifier_or_scalar_reuse(self):
        def bad(z):
            self.calls.append(z)
            return np.stack([self.image]*3)+(np.float32(.125) if len(self.calls)==2 else 0)
        with self.assertRaisesRegex(RuntimeError,'P RGB differs'):
            c.replay_source(self.ctx,0,self.pixels,latents=self.latents,decoder=bad,classifier=self.classifier)
        self.assertEqual(self.classifier.calls,[])

    def test_source_or_received_latent_corruption_never_repaired(self):
        self.pixels[0,0,0]=1
        with self.assertRaisesRegex(RuntimeError,'source pixels'):self.run_source()
        self.pixels[0,0,0]=0;self.z[0,0,0]=1
        with self.assertRaisesRegex(RuntimeError,'exact received latent'):self.run_source()
        self.assertEqual(self.calls,[])

    def test_classification_admission_and_software_failure_are_not_gray(self):
        self.classifier.identity['policy_sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'classifier identity'):self.run_source()
        self.classifier=FakeClassifier()
        def fail(_):raise RuntimeError('Synthetic resource failure')
        with self.assertRaisesRegex(RuntimeError,'resource failure'):
            c.replay_source(self.ctx,0,self.pixels,latents=self.latents,decoder=fail,classifier=self.classifier)
        self.assertEqual(self.classifier.calls,[])

    def test_no_clamp_and_no_seed_relabel_or_duplicate_context(self):
        self.image[0,0,0]=np.float32(1.000001)
        with self.assertRaisesRegex(RuntimeError,'no clamp'):self.run_source()
        docs=copy.deepcopy(self.docs);docs[1]['noise_seeds']=[6201,6202,6203]
        with self.assertRaisesRegex(RuntimeError,'latent schema'):c.context(*docs,inventory_sha256='e'*64)
        docs=copy.deepcopy(self.docs);docs[1]['selected_rows'][-1]=docs[1]['selected_rows'][-2]
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):c.context(*docs,inventory_sha256='e'*64)

    def test_final600_grid_preservesoldscalar_values_and_newproofs(self):
        one=self.run_source()['rows'];rows=[]
        for i in range(100):
            for row in one:
                new=copy.deepcopy(row);k=i,int(row['snr_db']),int(row['noise_seed']);old=self.ctx['scalar'][k]
                new.update(old);new['p_replay_original_scalar_sha256']=c.identity(old)
                new['p_replay_original_tensor_index']=self.ctx['selected'][k]['original_tensor_index'];rows.append(new)
        done=c.validate_complete(self.ctx,rows);self.assertEqual(done['frame_count'],600)
        with self.assertRaisesRegex(RuntimeError,'Exactly600'):c.validate_complete(self.ctx,rows[:-1])
        rows[0]['psnr_db']='13.0'
        with self.assertRaisesRegex(RuntimeError,'overwritten'):c.validate_complete(self.ctx,rows)

    def test_latent_batch_preserves_actual_index_order_and_byte_hash(self):
        # Sparse array-like proves original indices rather than assumed sequential slots.
        class Sparse:
            def __getitem__(s,j):
                self.calls.append(j);return self.z.copy()
        got=c.latent_batch(self.ctx,Sparse(),0,13)
        self.assertEqual(self.calls,[9,10,11]);self.assertEqual(got.shape,(3,32,16,16))
        self.ctx['selected'][0,13,2001]['Z_sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'byte identity'):c.latent_batch(self.ctx,Sparse(),0,13)

    def test_actual_downloaded_documents_admit_without_tensor_model_or_images(self):
        read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
        manifest=os.environ.get('P600_TEST_METADATA');pin=os.environ.get('P600_TEST_METADATA_SHA256')
        if manifest or pin:
            self.assertTrue(manifest and pin,'Both actual metadata fixture path and SHA are required')
            self.assertEqual(c.sha(manifest),pin)
            spec=read(manifest)
            self.assertEqual(spec['status'],'P600_CPU_TEST_METADATA_BOUND')
            paths=spec['paths']
            self.assertEqual(set(paths),{'latent_completion','latent_inventory','scalar_inventory',
                                        'metrics_registration','numerical_reference'})
            self.assertEqual(set(spec['input_bindings']),set(paths.values()))
            for p,s in spec['input_bindings'].items():
                self.assertTrue(Path(p).is_absolute());self.assertEqual(c.sha(p),s)
        else:
            base=Path(__file__).absolute().parent.parent/'current/P_reuse'
            self.assertTrue(base.exists(),'Actual metadata fixture required; no skip on production qualification')
            paths=dict(latent_completion=str(base/'latent_audit/completion.json'),
                latent_inventory=str(base/'latent_audit/latent_inventory.json'),
                scalar_inventory=str(base.parent/'h18_render_r2/prepared_v1/P_existing600_audit_v1/scalar_inventory.json'),
                metrics_registration=str(base/'results/unified_metrics_20261002/metrics_registration.json'),
                numerical_reference=str(base.parent/'original_native_qualification.json'))
        ctx=c.context(*(read(paths[k]) for k in ('latent_completion','latent_inventory','scalar_inventory',
            'metrics_registration','numerical_reference')),inventory_sha256=c.sha(paths['latent_inventory']))
        self.assertEqual(len(ctx['selected']),600);self.assertEqual(ctx['flags'],c.FLAGS)
        self.assertEqual(ctx['selected'][0,13,2001]['original_tensor_index'],9)

    def test_native_direct_Dc_adapter_preserves_B3_and_flags_without_channel(self):
        calls=[]
        class Tensor:
            def __init__(s,a):s.a=a;s.dtype=a.dtype;s.shape=a.shape
            def to(s,device):calls.append(('to',str(device)));return s
            def detach(s):return s
            def cpu(s):return s
            def numpy(s):return s.a
        class Decoder:
            training=False
            def parameters(s):return []
            def __call__(s,x):calls.append(('Dc',x.shape));return Tensor(np.stack([self.image]*3))
        torch=SimpleNamespace(__version__='fake',version=SimpleNamespace(cuda='fake'),float32=np.dtype('float32'),
            from_numpy=Tensor,no_grad=nullcontext,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=False,benchmark=False,deterministic=True)),
            get_float32_matmul_precision=lambda:'highest',are_deterministic_algorithms_enabled=lambda:True,
            get_num_threads=lambda:6,get_num_interop_threads=lambda:2)
        native=SimpleNamespace(torch=torch,loaded=dict(identity=self.ctx['native_identity'],device='cuda:0',decoder=Decoder()),
            frozen=lambda:calls.append(('frozen',)))
        decoder=c.FrozenDc(native,self.ctx);out=decoder(np.stack([self.z]*3));decoder.finish()
        np.testing.assert_array_equal(out,np.stack([self.image]*3))
        self.assertEqual(calls,[('frozen',),('to','cuda:0'),('Dc',(3,32,16,16)),('frozen',)])
        torch.backends.cuda.matmul.allow_tf32=True
        with self.assertRaisesRegex(RuntimeError,'flags changed'):decoder(np.stack([self.z]*3))

    def test_sealed_archive_load_is_CPU_weights_only_and_original_metadata(self):
        class Tensor:
            device=SimpleNamespace(type='cpu')
            def __init__(s,a):s.a=a;s.dtype=a.dtype
            def numpy(s):return s.a
        inv=self.ctx['inventory']
        z=np.zeros((1500,32,16,16),np.float32)
        data=dict(Z=Tensor(z),source_indices=Tensor(np.array([r['source_index'] for r in inv['channel_rows']],np.int64)),
            identity=inv['identity'],rows=inv['channel_rows'])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'fake_trusted_tensor.pt';path.write_bytes(b'CPU fixture no actual torch tensor')
            inv['original_tensor_path']=str(path);inv['original_tensor_sha256']=c.sha(path);inv['input_bindings'][str(path)]=c.sha(path)
            calls=[]
            def load(p,**kwargs):calls.append((p,kwargs));return data
            torch=SimpleNamespace(load=load,Tensor=Tensor,float32=np.dtype('float32'),int64=np.dtype('int64'))
            lat=c.SealedLatents(self.ctx,torch);self.assertEqual(lat.batch(0,13).shape,(3,32,16,16))
            self.assertEqual(calls,[(path,dict(map_location='cpu',weights_only=True))])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'tensor changed'):c.SealedLatents(self.ctx,torch)


if __name__=='__main__':unittest.main()
