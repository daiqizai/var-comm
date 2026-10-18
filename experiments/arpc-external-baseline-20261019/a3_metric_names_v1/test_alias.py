"""Synthetic alias regression; no actual dataset, model, GPU, score or RNG calls."""
import csv
import importlib.abc
import os
import pathlib
import sys
import tempfile
import unittest
from adapter import adapted

blocked=[]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','tensorflow','transformers','infinity'}:
            blocked.append(fullname)
            raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard())
E=pathlib.Path(__file__).parent.parent
sys.path.insert(0,str(E/'a3_metrics_v1'))
from data import METRICS,BASELINES

class Alias(unittest.TestCase):
    def test_changed_input_rejected(self):
        with self.assertRaises(AssertionError):adapted(b'changed')
    def test_exact_main_alias_and_values(self):
        base=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_external_20261019_v1/a3_metric_names_qualify_control_v1')
        with tempfile.TemporaryDirectory(dir=base) as directory:
            root=pathlib.Path(directory);paths={k:str(root/(k+'.csv')) for k in ['main','supplement','t1c']}
            def write(name,rows):
                with open(paths[name],'w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            main=[{'method':('BPG_ADAPTIVE' if m==BASELINES[-1] else m),'snr_db':s,'metric':k,'source_index':i,'source_id':str(i),'mean':-0.25,'noise_count':3} for m in BASELINES for s in [4,10,19] for k in METRICS[:4] for i in range(100)]
            write('main',main)
            write('supplement',[{'method':m,'snr_db':s,'metric':k,'source_index':i,'source_mean':.75,'noise_count':3} for m in BASELINES for s in [4,10,19] for k in METRICS[4:] for i in range(100)])
            write('t1c',[{'population':'common500','source_index':i,'family':'PARTIAL','metric':k,'new_mean':2} for i in range(100) for k in METRICS])
            sources=[{'source_index':i,'source_id':str(i)} for i in range(100)]
            fn=adapted((E/'a3_metrics_v1/data.py').read_bytes());out=fn(paths,sources)
            self.assertEqual(len(out),8400)
            for r in out:
                self.assertEqual(r['source_mean'],2 if r['method']=='EC_VAR_PARTIAL' and r['snr_db']==19 else (-.25 if r['metric'] in METRICS[:4] else .75))
            write('main',main+[{**main[-1],'method':BASELINES[-1]}])
            with self.assertRaises(AssertionError):fn(paths,sources)
            write('main',main[:-1])
            with self.assertRaises(AssertionError):fn(paths,sources)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Alias))
    print({'tests':result.testsRun,'blocked':blocked,'science':0})
    assert result.wasSuccessful() and result.testsRun==2 and not blocked
