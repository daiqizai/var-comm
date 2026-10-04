import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
extract=module('extract_headline');render=module('render_headline')
def require(value,message):
    if not value:raise RuntimeError(message)

class ScopeTests(unittest.TestCase):
    def test_only_original_selected_whole16qam_admitted(self):
        good=dict(experiment='M1',method='entropy_policy',phy_family='16QAM',action_id='N1024/16QAM/m8/K0/whole',action_m='8',action_q='0',order='whole')
        extract.validate_m1(good,require)
        for name,value in [('phy_family','QPSK'),('action_q','1'),('order','entropy'),('method','whole_policy')]:
            with self.subTest(name=name),self.assertRaises(RuntimeError):extract.validate_m1(dict(good,**{name:value}),require)
    def test_renderer_rejects_missing_image_and_wrong_scope(self):
        class Helper:pass
        m=Helper();m.require=require
        manifest=dict(status='HEADLINE_EXACT_CACHE_EXTRACTION_COMPLETE',method_labels={'M1':render.M1_LABEL})
        assets={i:dict(status='AVAILABLE') for i in range(80)}
        rows={i:dict(method='M1',action_id='N1024/16QAM/m8/K0/whole',phy_family='16QAM',E='2124.8') for i in range(64)}
        render.validate_scope(m,manifest,[(1024,13)],assets,rows)
        assets[0]['status']='UNAVAILABLE_CACHE'
        with self.assertRaises(RuntimeError):render.validate_scope(m,manifest,[(1024,13)],assets,rows)
        assets[0]['status']='AVAILABLE'
        with self.assertRaises(RuntimeError):render.validate_scope(m,manifest,[(1024,7)],assets,rows)
    def test_renderer_requires_real_energy(self):
        class Helper:pass
        m=Helper();m.require=require
        manifest=dict(status='HEADLINE_EXACT_CACHE_EXTRACTION_COMPLETE',method_labels={'M1':render.M1_LABEL})
        assets={i:dict(status='AVAILABLE') for i in range(80)}
        rows={i:dict(method='M1',action_id='N1024/16QAM/m8/K0/whole',phy_family='16QAM',E='0') for i in range(64)}
        with self.assertRaises(RuntimeError):render.validate_scope(m,manifest,[(1024,13)],assets,rows)

if __name__=='__main__':unittest.main()
