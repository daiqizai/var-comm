import pathlib,unittest
import native
from cache import match
class ActualRX(unittest.TestCase):
    def state(self):return {'generation_seed':0,'endpoint':4,'wire':{'sha256':'abc','bytes':50},'image':{'path':'independent_rx_image'}}
    def job(self):return {'key':'k','generation_seed':0,'paid_profile':4,'input':{'sha256':'abc','bytes':50}}
    def test_exact_payload_cache(self):
        a,b=match([self.job()],[self.state()]);self.assertEqual(len(a),1);self.assertFalse(b)
    def test_received_corruption_does_not_use_truth(self):
        j=self.job();j['input']['sha256']='received_wrong';a,b=match([j],[self.state()]);self.assertFalse(a);self.assertEqual(b,[j])
    def test_paid_endpoint_changes_cache(self):
        j=self.job();j['paid_profile']=3;self.assertEqual(len(match([j],[self.state()])[1]),1)
    def test_seed_cannot_change(self):
        j=self.job();j['generation_seed']=1
        with self.assertRaises(AssertionError):match([j],[self.state()])
    def test_duplicate_jobs_rejected(self):
        with self.assertRaises(AssertionError):match([self.job(),self.job()],[self.state()])
    def test_unique_state_limit(self):
        jobs=[dict(self.job(),key=str(i)) for i in range(6301)]
        with self.assertRaises(AssertionError):match(jobs,[])
    def test_original_engine_pin(self):
        p=pathlib.Path(__file__).parents[1]/'a1_native.py';result=native.adapt(p.read_bytes())
        self.assertIn('None,trans,modes)',result);self.assertIn('except SourceParseError',result)
    def test_engine_mutation_rejected(self):
        with self.assertRaises(AssertionError):native.adapt(b'changed engine')
if __name__=='__main__':unittest.main(verbosity=2)
