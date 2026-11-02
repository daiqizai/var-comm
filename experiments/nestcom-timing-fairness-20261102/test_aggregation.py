"""CPU-only tests of source weighting and rejection of incomplete timing cells."""
import unittest
from summarize import aggregate,cpu_telemetry

class AggregationTests(unittest.TestCase):
    def test_source_average_and_total(self):
        data=[]
        for source,offset in [('a',0),('b',10)]:
            for direction,scale in [('tx',1),('rx',2)]:
                data.append(dict(method='m',source_id=source,direction=direction,phase='warmup',seconds=999))
                data.extend(dict(method='m',source_id=source,direction=direction,phase='measured',seconds=scale*(offset+v))for v in [1,2,3])
        got=aggregate(data)
        self.assertEqual(got['m','a','tx'],2)
        self.assertEqual(got['m','b','total'],36)
        self.assertEqual((got['m','a','total']+got['m','b','total'])/2,21)
    def test_incomplete_cell_is_not_averaged(self):
        with self.assertRaises(AssertionError):
            aggregate([dict(method='m',source_id='a',direction='tx',phase='measured',seconds=1)]*2)
    def test_tx_only_grid_does_not_invent_rx(self):
        got=aggregate([dict(method='grid',source_id='a',direction='tx',phase='measured',seconds=1)]*3)
        self.assertEqual(set(got),{('grid','a','tx')})
    def test_telemetry_uses_deltas_and_excludes_iowait(self):
        snapshots=[dict(cpu=['cpu38 1 2 3 4 5 6 7 8']),dict(cpu=['cpu38 2 2 3 6 6 6 7 8'])]
        self.assertEqual(cpu_telemetry(snapshots),{'cpu38':.25})

if __name__=='__main__':unittest.main(verbosity=2)
