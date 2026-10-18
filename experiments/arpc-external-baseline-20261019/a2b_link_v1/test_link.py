import json,pathlib,tempfile,unittest
import worker
from plan import events
def fixture():
    sources=[]
    for i in range(2):sources.append({'source_id':str(i),'source_index':i,'streams':[{'endpoint':e,'payload_bits':e*300,'wire':{'sha256':f'{i}-{e}'}} for e in range(1,8)]})
    d={s:{'proposed_candidates':v} for s,v in [('4',['e7_q2_r2-3','e7_q2_r1-2']),('10',['e7_q4_r2-3','e7_q4_r3-4']),('19',['e7_q6_r5-6','e7_q4_r2-3'])]}
    return sources,d,worker.link.templates()
class Grid(unittest.TestCase):
    def test_logical_grid_denominator(self):
        self.assertEqual(len(events(*fixture())),36)
    def test_same_waveform_reused_but_noise_separate(self):
        a=events(*fixture());x=[r for r in a if r['source_index']==0 and r['candidate_id']=='e7_q4_r2-3'];self.assertEqual(len(x),6)
        self.assertEqual(len({r['waveform_key'] for r in x}),1);self.assertEqual(len({r['key'] for r in x}),6)
    def test_actual_prefix_uses_paid_capacity(self):
        a=events(*fixture());x=next(r for r in a if r['candidate_id']=='e7_q2_r1-2');self.assertEqual(x['endpoint'],3)
    def test_no_fit_keeps_full_denominator(self):
        s,d,p=fixture()
        for r in s:
            for x in r['streams']:x['payload_bits']=10000
        rows=events(s,d,p);self.assertEqual(len(rows),36);self.assertTrue(all(r['source_no_fit'] and r['wire'] is None for r in rows))
    def test_duplicate_sources_stop(self):
        s,d,p=fixture();s[1]['source_id']=s[0]['source_id']
        with self.assertRaises(AssertionError):events(s,d,p)
    def test_unresolved_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'ledger';p.write_text(json.dumps({'event':1,'state':'RESERVED','kind':'body'})+'\n')
            with self.assertRaises(AssertionError):worker.ledger_closed(p,{'body':1})
if __name__=='__main__':unittest.main(verbosity=2)
