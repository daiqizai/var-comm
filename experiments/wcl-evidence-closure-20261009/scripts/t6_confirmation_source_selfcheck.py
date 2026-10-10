"""Synthetic100 owner closure/duplicate gating; no image model or channel calls."""
import argparse,hashlib,os,sys,tempfile,time,unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import t6_confirmation_sources as owner
from t1_entropy_core import OFFSETS,read,sha

class Tensor:
    def __init__(self,a):self.a=a
    def cpu(self):return self
    def numpy(self):return self.a

class ToyCodec:
    def __init__(self,*a,**k):pass
    def encode(self,family,tokens,modes):
        return {m:dict(bits=np.unpackbits(tokens[:OFFSETS[m]].astype('<u2').view(np.uint8)),raw_bits=int(12*OFFSETS[m]),
            arithmetic_bits=int(16*OFFSETS[m]),flush_bits=2) for m in modes}
    def decode(self,family,bits,m):
        return dict(received_tokens=np.packbits(bits).view('<u2').astype(np.int64),canonical=True,zero_extension_reads=30)

def forbidden():raise AssertionError('forbidden science stub invoked')
def harmless():pass

class Checks(unittest.TestCase):
    def fixture(self,root,duplicate=False):
        out=root/'run';out.mkdir();images=root/'images';images.mkdir();records=[]
        for i in range(100):
            p=images/f'{i:04d}.JPEG';p.write_bytes(('SYNTHETIC_BYTES_'+str(i)).encode())
            records.append(dict(source_index=i,source_id='synthetic:'+str(i),evaluation_class_index=i,path=str(p),original_bytes=p.stat().st_size))
        def doc(name,value):
            p=root/name;owner.save(p,value);return owner.pin(p)
        registration=doc('registration.json',dict(image_root=str(images),records=records))
        reference=doc('reference.json',dict(records=[dict(raw_JPEG_sha256=[sha(images/'0000.JPEG')] if duplicate else [],preprocessing_sha256=[])],
            available_reference_hash_count=1,unavailable_reference_hash_count=0,source_without_raw_JPEG_hash=0,source_without_preprocessing_hash=1))
        template=doc('template.json',{'python':os.path.abspath(sys.executable)});freeze=doc('freeze.json',{'selection_used_confirmation':False})
        selected=doc('selected.json',{'holdout_files_read':False});env=doc('env.json',{'visual_config':{'visual_lock':str(root/'visual.lock')}})
        static=doc('static.json',{'synthetic':True});module=doc('module.json',{'synthetic':True})
        policies={str(s):dict(candidate_id='synthetic-'+str(s),target_m=m,q=q,nominal_rate='2/3',source_capacity=c)
            for s,m,q,c in [(4,7,2,1951),(10,9,4,5251),(19,10,6,7891)]}
        request=dict(schema='T6_CONFIRMATION100_SOURCE_REQUEST_V1',root=str(root),out=str(out),source_bindings={},records=records,source_ids=[r['source_id'] for r in records],
            confirmation_registration=registration,calibration_freeze=freeze,entropy_family_selection=selected,reference_hash_inventory=reference,
            original_source_request=template,environment_request=env,static_completion=static,preprocess_module=module,original_encoder_module=module,
            max_Encoder_VQ_calls=100,max_VAR_TX_calls=100,max_VAR_RX_calls=300,deadline_unix=time.time()+60,max_seconds=60,stop_files=[],
            entropy_policies=policies,family='EC_VAR_WHOLE')
        path=out/'request.json';owner.save(path,request);return path

    def runtime(self):
        native=SimpleNamespace(health_polling=SimpleNamespace(next_check=0),common=SimpleNamespace(check=harmless,quality=forbidden),
            torch=SimpleNamespace(no_grad=nullcontext),data=forbidden,score=forbidden,qualify=forbidden,
            phy=SimpleNamespace(receive=forbidden),receiver=SimpleNamespace(complete_partial=forbidden),frozen=harmless)
        def preprocess(path):
            i=int(Path(path).stem);p=np.full((3,256,256),i,np.uint8)
            return Tensor(p.astype(np.float32)/np.float32(127.5)-np.float32(1)),hashlib.sha256(p.tobytes()).hexdigest()
        def encode(native,pixels):return np.full(680,int(pixels[0,0,0]),np.int64),np.zeros((32,16,16),np.float32)
        def loader(desc,name):return SimpleNamespace(preprocess=preprocess) if 'preprocess' in name else SimpleNamespace(encode_flat=encode)
        return native,loader

    def execute(self,path):
        native,loader=self.runtime()
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),\
            patch.object(os,'sched_getaffinity',lambda *a:set(range(4,10)),create=True),\
            patch.object(os,'getpriority',lambda *a:15,create=True),patch.object(os,'PRIO_PROCESS',0,create=True),\
            patch.object(owner.shared,'lock',lambda *a:nullcontext()),patch.object(owner.shared,'add_paths',lambda *a:None),\
            patch.object(owner,'module',loader),patch.object(owner,'build_native',lambda *a:(native,None)),\
            patch.object(owner,'SourceCodec10',ToyCodec),patch('builtins.print'):
            return owner.run(str(path))

    def test_synthetic100_closes_exact_counts_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(Path(tmp));result=self.execute(path)
            self.assertEqual(result['counts'],dict(Encoder_VQ=100,VAR_TX=100,VAR_RX=300))
            d=read(path.parent/'manifest.json');self.assertEqual(d['schema'],'T6_CONFIRMATION100_SOURCE_MANIFEST_V1')
            self.assertEqual(len(d['records']),100);self.assertTrue(all('entropy_checkpoint' in r for r in d['records']))
            self.assertEqual(read(path.parent/'content_duplicate_check.json')['duplicate_preprocessing_count'],0)

    def test_duplicate_stops_before_any_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(Path(tmp),duplicate=True)
            with self.assertRaisesRegex(ValueError,'Content duplicate found'):self.execute(path)
            d=read(path.parent/'owner_failed.json');self.assertEqual(d['counts'],dict(Encoder_VQ=0,VAR_TX=0,VAR_RX=0))
            self.assertFalse((path.parent/'completion.json').exists())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    owner.save(a.out,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,synthetic_only=True,
        real_image_decodes=0,real_Encoder_calls=0,real_VAR_calls=0,real_channel_calls=0,
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ('t6_confirmation_sources.py','t6_confirmation_source_selfcheck.py')}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
