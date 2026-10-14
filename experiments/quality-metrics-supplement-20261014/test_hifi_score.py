"""Synthetic grid guards only: no models, real pixels, channel, or scores."""
import copy,importlib.util,sys,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('score',Path(__file__).with_name('hifi_score.py'));s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
def fixture():
    sources=[dict(source_index=i,source_id='fixture/'+str(i),preprocessing_id='f'*64) for i in s.INDICES]
    rows=[]
    for src in sources:
        for method in s.METHODS:
            row=dict(method=method,source_index=src['source_index'],source_id=src['source_id'],source_pixels_sha256=src['preprocessing_id'],N=1024,snr_db=13,noise_seed=2001,archive=dict(path='fixture/'+method+str(src['source_index']),sha256='a'*64),checkpoint_sha256='c'*64,observation_sha256='d'*64)
            if method=='HiFiSwin200000':row['paired_Swin_archive']=dict(path='fixture/Swin200000'+str(src['source_index']),sha256='a'*64)
            rows.append(row)
    return rows,sources
class Tests(unittest.TestCase):
    def test_complete16(self):
        rows,sources=fixture();self.assertEqual(len(s.validate_grid(rows,sources)),48)
    def test_missing16th(self):
        rows,sources=fixture()
        with self.assertRaises(RuntimeError):s.validate_grid(rows[:-1],sources)
    def test_no_19db_substitution(self):
        rows,sources=fixture();rows[0]['snr_db']=19
        with self.assertRaises(RuntimeError):s.validate_grid(rows,sources)
    def test_exact_observation_required(self):
        rows,sources=fixture();rows[0]['observation_sha256']='e'*64
        with self.assertRaises(RuntimeError):s.validate_grid(rows,sources)
    def test_duplicated_source(self):
        rows,sources=fixture();rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaises(RuntimeError):s.validate_grid(rows,sources)
if __name__=='__main__':unittest.main()
