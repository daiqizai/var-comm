"""Synthetic selection-scope checks for full calibration; no science runs."""
import tempfile,unittest
from pathlib import Path
from t2_recheck import build_schedule,validate,pilot_checkpoint,h
from t2_gpu_parallel import assignments,validate_config

def fixture():
    profiles=[dict(candidate_id=f'w{i:03d}',wire_key=f'wire{i}',K=0,N=1024,allocation_modes=['full_budget']if i>=99 else['nominal'])for i in range(106)]
    profiles.append(dict(candidate_id='p',wire_key='partial',K=142,N=1024,allocation_modes=['nominal']))
    ranks=[dict(snr_db=s,pilot_rank=i+1,candidate_id=f'w{i:03d}')for s in [10,19]for i in range(106)]
    policy=dict(winners=[dict(snr_db=s,family=f,candidate_id=c)for s in [10,19]for f,c in [('WHOLE','w050'),('PARTIAL','p')]])
    return profiles,ranks,policy

class FullScope(unittest.TestCase):
    def test_all_protection_and_both_original_winners_retained(self):
        p,r,o=fixture();s=build_schedule(r,p,o)
        self.assertEqual(len(s),28)
        for snr in [10,19]:
            cells=[x for x in s if x['snr_db']==snr]
            self.assertEqual(sum('full_budget_whole'in x['reasons']for x in cells),7)
            self.assertEqual(sum('original_partial_winner'in x['reasons']for x in cells),1)
            self.assertEqual(sum('original_whole_winner'in x['reasons']for x in cells),1)
            self.assertFalse(next(x for x in cells if x['candidate_id']=='p')['eligible_for_expanded_whole'])
    def test_overlapping_actions_deduplicated(self):
        p,r,o=fixture()
        for w in o['winners']:w['candidate_id']='w000'
        s=build_schedule(r,p,o);self.assertEqual(len(s),24)
        row=next(x for x in s if x['candidate_id']=='w000')
        self.assertIn('pilot_top5',row['reasons']);self.assertIn('original_partial_winner',row['reasons'])
    def test_missing_pilot_action_refused(self):
        p,r,o=fixture();r.pop()
        with self.assertRaisesRegex(RuntimeError,'All pilot ranks'):build_schedule(r,p,o)
    def test_repeated_candidate_cannot_hide_missing_action(self):
        p,r,o=fixture();r[1]['candidate_id']=r[0]['candidate_id']
        with self.assertRaisesRegex(RuntimeError,'unique'):build_schedule(r,p,o)
    def test_missing_strong_control_refused(self):
        p,r,o=fixture();p[105]['allocation_modes']=['nominal']
        with self.assertRaisesRegex(RuntimeError,'seven'):build_schedule(r,p,o)
    def test_budget_and_noise_cannot_expand(self):
        p,rank,o=fixture();s=build_schedule(rank,p,o)
        r=dict(schema='WCL_T2_FULL1000_RECHECK_V1',source_count=1000,noise_seeds=[4101,4102,4103],snrs=[10,19],records=[dict(source_index=i)for i in range(1000)],workers=8,schedule=s,frame_count=84000,packet_cap=168000,root='/repo',out='/repo/outputs/WCL_FULL')
        validate(r);r['noise_seeds'].append(4104)
        with self.assertRaisesRegex(RuntimeError,'scope'):validate(r)
    def test_parent_source_must_match_complete_pilot_seal(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'cpu_sources'/'0000.json';h.save(p,dict(request_sha256='bound',frames=[]))
            pilot=dict(out=tmp);done=dict(outputs={str(p):h.sha(p)})
            self.assertEqual(pilot_checkpoint(pilot,done,'cpu_sources',0,'bound')['frames'],[])
            p.write_text('{"request_sha256":"bound","frames":[1]}')
            with self.assertRaisesRegex(RuntimeError,'parent pilot output changed'):pilot_checkpoint(pilot,done,'cpu_sources',0,'bound')
    def test_gpu_shards_cover_original_sources_exactly_once(self):
        a,b=assignments();self.assertEqual(sorted(a+b),list(range(1000)));self.assertFalse(set(a)&set(b))
    def test_cohort_cannot_change_source_assignment(self):
        r=dict(gpu_workers=2,out='/out');s=Path('/out/gpu_cohorts/a')
        c=dict(request_sha256='request',worker_id=0,stage='production',source_indices=assignments()[0],session=str(s),cohort_manifest=str(s/'cohort.json'),stop_file=str(s/'STOP'))
        validate_config(r,'request',c);c['source_indices']=[0,1]
        with self.assertRaisesRegex(RuntimeError,'disjoint'):validate_config(r,'request',c)

if __name__=='__main__':unittest.main()
