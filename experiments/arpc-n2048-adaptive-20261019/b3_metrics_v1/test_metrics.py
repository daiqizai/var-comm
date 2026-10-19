"""Synthetic source-paired statistics and AST scope proof; no models or science."""
from pathlib import Path
import ast, copy, importlib.abc, sys, unittest


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','tensorflow','sionna','transformers','timm','lpips','dreamsim'}:
            raise AssertionError('Synthetic test forbids model import: '+fullname)
        return None


sys.meta_path.insert(0, Guard())
import numpy as np
import adapter
from statistical import *
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[1]/'arpc-n2048-validation-20261019/metrics.py'


def fixture():
    return [dict(method=m,source_id=s,snr_db=n,noise_seed=z,
        **{k:(i+1)*10+(z-6201)+(0 if m==METHODS[0] else i-1)for k in METRICS})
        for m in METHODS for i,s in enumerate(['a','b','c'])for n in SNRS for z in SEEDS]


class Tests(unittest.TestCase):
    def test_source_denominator_and_mean(self):
        r=source_means(fixture(),['a','b','c'])
        self.assertEqual(len(r),84);self.assertEqual(r[0]['source_mean'],11)

    def test_missing_frame_rejected(self):
        with self.assertRaises(AssertionError):source_means(fixture()[:-1],['a','b','c'])

    def test_duplicate_frame_rejected(self):
        a=fixture();a[-1]=a[0]
        with self.assertRaises(AssertionError):source_means(a,['a','b','c'])

    def test_no_fixed19_or_calibration_noise(self):
        for key,value in [('snr_db',19),('noise_seed',4101),('method','NEST_FULL_M10_N2048')]:
            a=fixture();a[0][key]=value
            with self.assertRaises(AssertionError):source_means(a,['a','b','c'])

    def test_failure_gray_keeps_full_source_denominator(self):
        a=fixture()
        for r in a:
            if r['source_id']=='a':r.update({k:0 for k in METRICS})
        r=source_means(a,['a','b','c']);self.assertEqual(r[0]['source_mean'],0)
        self.assertEqual(r[0]['noise_count'],3);self.assertEqual(len(r),84)

    def test_direct_paired_intervals_not_marginal_subtraction(self):
        draws=np.array([[0,0,0],[1,1,1],[2,2,2],[0,1,2]],np.int64)
        s,p=summarize(source_means(fixture(),['a','b','c']),['a','b','c'],draws)
        self.assertEqual(len(s),28);self.assertEqual(len(p),14)
        self.assertEqual(p[0]['mean'],0);self.assertTrue(p[0]['interval_crosses_or_touches_zero'])
        self.assertAlmostEqual(p[0]['ci_low'],-.925);self.assertAlmostEqual(p[0]['ci_high'],.925)
        self.assertNotAlmostEqual(p[0]['ci_low'],s[0]['ci_low']-s[1]['ci_high'])

    def test_negative_differences_preserved(self):
        a=fixture()
        for r in a:
            if r['method']==METHODS[0]:r.update({k:-10 for k in METRICS})
        _,p=summarize(source_means(a,['a','b','c']),['a','b','c'],np.array([[0,1,2]],np.int64))
        self.assertLess(p[0]['mean'],0)

    def test_nonfinite_rejected(self):
        a=fixture();a[0]['dreamsim']=float('nan')
        with self.assertRaises(AssertionError):source_means(a,['a','b','c'])

    def test_invalid_bootstrap_source_index_rejected(self):
        with self.assertRaises(AssertionError):summarize(source_means(fixture(),['a','b','c']),['a','b','c'],np.array([[0,1,3]],np.int64))

    def test_pinned_original_and_inverse_ast_proof(self):
        b=ORIGINAL.read_bytes();original=adapter.selected_tree(b);new=adapter.adapt(b,14400)
        self.assertEqual(ast.dump(ast.Module(body=original.body[:-1],type_ignores=[])),
                         ast.dump(ast.Module(body=new.body[:-1],type_ignores=[])))
        values=[n.value for n in ast.walk(new.body[-1]) if isinstance(n,ast.Constant)]
        self.assertEqual(values.count(1200),2);self.assertEqual(values.count(14340),1)
        self.assertNotIn(1800,values)

    def test_changed_original_source_rejected(self):
        with self.assertRaises(AssertionError):adapter.adapt(ORIGINAL.read_bytes()+b'\n',14400)

    def test_no_real_runtime_imported(self):
        self.assertFalse({'torch','tensorflow','sionna','transformers','dreamsim'} & set(sys.modules))

    def test_zero_missing_reuse_schema_and_no_model_calls(self):
        # Extract only the zero-new CPU branch; importing worker/common would
        # unnecessarily resolve production paths. All I/O stays in this fixture.
        import csv, hashlib, json, math, os, tempfile
        tree=ast.parse((HERE/'worker.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reuse_only')
        before=set(sys.modules)
        with tempfile.TemporaryDirectory(prefix='b3_metrics_zero_new_') as tmp:
            root=Path(tmp);old=root/'prior';(old/'a3_metrics_v1').mkdir(parents=True)
            out=root/'out';out.mkdir()
            def pin(p):
                b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
            def verify(d):
                p=Path(d['path']);self.assertEqual(pin(p),d);return p
            def put(p,v):
                with p.open('x') as f:json.dump(v,f)
            def read(p):return json.loads(p.read_text())
            def csvwrite(p,rows):
                with p.open('x',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            originals={
                'four_metric_identity.json':b'{"identity":"fixture-four","metadata":{"FP32":true}}\n',
                'supplemental_identity.json':b'{"protocol":{"fixture":true},"FP32":true,"TF32":false}\n',
            }
            inputs=[]
            for name,data in originals.items():
                p=old/'a3_metrics_v1'/name;p.write_bytes(data);inputs.append(pin(p))
            plan=[];expected_keys=[]
            for i in range(100):
                sid=f'source{i}'
                key=hashlib.sha256(('original-metric-identity|'+sid).encode()).hexdigest();expected_keys.append(key)
                frames=[dict(key=f'{sid}|{m}|{s}|{z}',source_id=sid,method=m,snr_db=s,noise_seed=z,
                    receiver_status='CHANNEL_OR_SOURCE_NULL_GRAY',accepted_wrong_post_exit=False)
                    for m in METHODS for s in SNRS for z in SEEDS]
                values={k:float(j-2) for j,k in enumerate(METRICS)}
                plan.append(dict(source_id=sid,source_index=i,pairs=[dict(image_float_sha256='a'*64,
                    frames=frames,reuse=dict(scores=values,old_pair_key=key))]))
            put(out/'pair_plan.json',dict(sources=plan))
            registration=dict(new_seven_metric_pairs=0,unique_pairs=100,inputs=inputs,stage_seconds=dict(score=14400))
            installs=[]
            namespace=dict(Path=Path,hashlib=hashlib,math=math,os=os,METRICS=METRICS,MOUT=out,OLD_O=old,
                check=lambda expected:registration,install=lambda seconds:installs.append(seconds),
                read=read,put=put,pin=pin,verify=verify,csvwrite=csvwrite,sha=lambda p:pin(p)['sha256'])
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<zero-new-fixture>','exec'),namespace)
            namespace['reuse_only']('fixture-registration')
            self.assertEqual(installs,[14400])
            done=read(out/'score_completion.json')
            self.assertEqual((done['logical_frames'],done['unique_pairs'],done['new_pairs'],done['reused_pairs']),
                             (1200,100,0,100))
            self.assertEqual(done['ledger'],dict(counts=[],unresolved=0));self.assertEqual(done['bootstrap'],0)
            self.assertEqual({Path(d['path']).name for d in done['outputs']},
                {'logical_scores.csv','actual_pair_scores.csv','four_metric_identity.json','supplemental_identity.json'})
            for d in done['outputs']:verify(d)
            for name,data in originals.items():self.assertEqual((out/name).read_bytes(),data)
            with (out/'actual_pair_scores.csv').open() as f:pairs=list(csv.DictReader(f))
            with (out/'logical_scores.csv').open() as f:logical=list(csv.DictReader(f))
            self.assertEqual([p['pair_key'] for p in pairs],expected_keys)
            self.assertEqual(len(logical),1200)
            self.assertEqual(float(pairs[0]['psnr_db']),-2.0)
            self.assertEqual(float(pairs[0]['dinov2_vitl14_cosine']),0.0)
            self.assertTrue(all(r['receiver_status']=='CHANNEL_OR_SOURCE_NULL_GRAY' for r in logical))
            with self.assertRaises(FileExistsError):namespace['reuse_only']('fixture-registration')
        self.assertFalse({'torch','tensorflow','sionna','transformers','dreamsim','timm','lpips'} & (set(sys.modules)-before))


if __name__=='__main__':unittest.main(verbosity=2)
