import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import prepare_clean_missing_remote as b

c=b.c
HERE=Path(__file__).absolute().parent
def put(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return str(p)

class BuilderTests(unittest.TestCase):
    def model_fixture(self,root):
        root=Path(root);prior={};identity={'models':{},'quality':{'config':{}}}
        paths={}
        for key in ('vae_checkpoint','var_checkpoint','dino_checkpoint','alexnet_checkpoint'):
            p=root/(key+'.pth');p.write_bytes(key.encode());h=c.sha(p)
            if key in ('vae_checkpoint','var_checkpoint'):paths.update({key:str(p),key+'_sha256':h})
            else:identity['quality']['config'].update({key:str(p),key+'_sha256':h})
        yaml=root/'configs/next_scale_prior_diagnostic.yaml';yaml.parent.mkdir(parents=True)
        yaml.write_text('paths:\n'+''.join('  '+k+': '+v+'\n' for k,v in paths.items()));prior[str(yaml)]=c.sha(yaml)
        linear=root/'linear.pth';linear.write_bytes(b'linear');prior[str(linear)]=c.sha(linear);identity['quality']['linear_weights_sha256']=c.sha(linear)
        decoder=root/'decoder.pt';decoder.write_bytes(b'decoder');gate=put(root/'gate.json',{'selection':{'checkpoint':str(decoder),'checkpoint_sha256':c.sha(decoder)}})
        identity.update(decoder_gate=gate,decoder_gate_sha256=c.sha(gate))
        stats=root/'outputs/RX-POSTERIOR-STEP1-20260929/calibration_statistics.pt';stats.parent.mkdir(parents=True);stats.write_bytes(b'stats')
        identity['calibration_statistics_sha256']=c.sha(stats)
        cal=put(root/'cal.json',{'identity':identity});num=put(root/'native_q.json',{'frozen_identity':identity});prior[num]=c.sha(num)
        replay=root/'experiments/unified-metrics-20261002/replay.py';replay.parent.mkdir(parents=True);replay.write_text('# frozen')
        prior[str(replay)]=c.sha(replay)
        supervisor=put(root/'outputs/UNIFIED-METRICS-20261002/supervisor_registration.json',{'source_bindings':{str(replay):c.sha(replay)}})
        return dict(root=root,v={'native_qualification':num},prior=prior,oc={'calibration_registration':cal},oldprior={cal:c.sha(cal)}),supervisor

    def test_original_model_paths_and_unflattened_replay_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            x,supervisor=self.model_fixture(d);self.assertNotIn(supervisor,x['prior'])
            inputs,models=b.model_inputs(x);self.assertEqual(len(models),7);self.assertIn(supervisor,inputs)
            self.assertTrue(all(c.sha(p)==h for p,h in models.items()))
            replay=next(p for p in x['prior'] if p.endswith('replay.py'));Path(replay).write_text('# changed')
            with self.assertRaisesRegex(ValueError,'replay'):b.model_inputs(x)

    def test_model_file_mismatch_never_substitutes_state_digest(self):
        with tempfile.TemporaryDirectory() as d:
            x,_=self.model_fixture(d);(Path(d)/'var_checkpoint.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen model'):b.model_inputs(x)

    def run_qualification(self,d,code=0):
        root=Path(d);fixture=root/'actual_fixture.json';put(fixture,{'metadata_only':True})
        sources={str(HERE/n):c.sha(HERE/n) for n in b.SOURCES}
        x=dict(work=root/'qualification',fixtures={'fixture':str(fixture)},inputs={},sources=sources,runtime=HERE,v={'python':'/old/UM/python'})
        waited=[]
        class Child:
            pid=1234
            def __init__(self,*a,**kw):kw['stdout'].write('Ran 9 tests in 0.1s\n\n'+('OK' if code==0 else 'FAILED')+'\n')
            def wait(self):waited.append(True);return code
            def poll(self):return code
        with mock.patch.object(b.sys,'platform','linux'),mock.patch.object(b.subprocess,'Popen',Child),mock.patch.object(c,'proc',return_value={'pid':1234,'start_ticks':9,'uid':1002,'argv':['UM']}):
            if code:
                with self.assertRaisesRegex(ValueError,'qualification failed'):b.qualify(x)
            else:b.qualify(x)
        self.assertEqual(waited,[True]);return x

    def test_real_qualification_consumer_after_closed_waited_log(self):
        with tempfile.TemporaryDirectory() as d:
            x=self.run_qualification(d);p=str(x['work']/'completion.json')
            inputs={p:c.sha(p)};c.qualification({'python':'/old/UM/python','prepared_qualification':p},x['sources'],inputs)
            self.assertIn(str(x['work']/'tests.log'),inputs)
            with self.assertRaisesRegex(ValueError,'Fresh qualification'):b.qualify(x)

    def test_failed_tests_preserved_without_receipt_or_request(self):
        with tempfile.TemporaryDirectory() as d:
            x=self.run_qualification(d,1);self.assertTrue((x['work']/'failure.json').is_file())
            self.assertTrue((x['work']/'exit.json').is_file());self.assertFalse((x['work']/'completion.json').exists())

if __name__=='__main__':unittest.main()
