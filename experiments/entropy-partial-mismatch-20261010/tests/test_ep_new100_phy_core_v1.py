"""Metadata and synthetic arrays only: no noise generation, channel, or decoder."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import ep_new100_phy_core_v1 as c
import ep_phy
import t1_phy
from test_ep_new100_confirmation_core_v1 import fixture,grid


class FourArmPHY(unittest.TestCase):
    def test_complete_three_domains(self):
        rows=grid(fixture());c.validate_grid(rows)
        self.assertEqual({g:sum(c.group_for(x['method'])==g for x in rows) for g in c.GROUPS},dict(raw=1800,whole=900,partial=900))
    def test_missing_failure_row_cannot_close(self):
        rows=grid(fixture());rows.pop()
        with self.assertRaises(ValueError):c.validate_grid(rows)
    def test_reordered_logical_indices_rejected(self):
        rows=grid(fixture());rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError):c.validate_grid(rows)
    def test_old_whole_fallback_stays_K0(self):
        event=next(x for x in grid(fixture()) if x['method']=='EC_VAR_WHOLE');candidate=event['policy']
        streams={(m,K):np.zeros(2 if m==4 and K==0 else candidate['source_capacity_bits']+1,dtype=np.uint8)
            for m in range(4,10) for K in (0,)+c.plan.ks(m)}
        profiles={str(x['profile_id']):x for x in t1_phy.profile_templates()}
        profile,bits,selected=c.entropy_choice(event,streams,profiles,'whole')
        self.assertEqual((profile['m'],profile['K']),(4,0));self.assertEqual(len(bits),2)
        self.assertTrue(all(x['K']==0 for x in selected['attempts']))
    def test_old_whole_cannot_receive_with_EP360(self):
        event=next(x for x in grid(fixture()) if x['method']=='EC_VAR_WHOLE')
        profiles={str(x['profile_id']):x for x in ep_phy.profile_templates()}
        with self.assertRaisesRegex(ValueError,'full entropy receive domain'):c.entropy_choice(event,{},profiles,'whole')
    def test_missing_actual_stream_not_fabricated_gray(self):
        event=next(x for x in grid(fixture()) if x['method']=='EC_VAR_WHOLE')
        profiles={str(x['profile_id']):x for x in t1_phy.profile_templates()}
        with self.assertRaisesRegex(ValueError,'Missing actual encoded stream'):c.entropy_choice(event,{},profiles,'whole')
    def test_no_fitting_actual_stream_is_software_stop(self):
        event=next(x for x in grid(fixture()) if x['method']=='EC_VAR_WHOLE');capacity=event['policy']['source_capacity_bits']
        streams={(m,0):np.zeros(capacity+1,dtype=np.uint8) for m in range(4,10)}
        with self.assertRaisesRegex(ValueError,'cannot fabricate channel failure'):
            c.entropy_choice(event,streams,{str(x['profile_id']):x for x in t1_phy.profile_templates()},'whole')
    def test_domains_prevent_reuse_even_when_actual_arrays_equal(self):
        event=next(x for x in grid(fixture()) if x['method']=='EC_VAR_WHOLE')
        item=dict(tokens_sha256='a'*64,source_assets={'sha256':'b'*64});zeros=np.zeros((1024,2),dtype=np.float64)
        array_sha=lambda a:hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
        args=(event,item,'whole',{'profile_id':72},np.zeros(2,dtype=np.uint8),zeros,zeros,zeros,dict(group='whole',profile_count=144),array_sha)
        old=c.identity(*args)
        args=list(args);args[2]='partial';args[-2]=dict(group='partial',profile_count=360);new=c.identity(*args)
        self.assertEqual(old['observation_sha256'],new['observation_sha256']);self.assertNotEqual(c.digest(old),c.digest(new))
    def results(self):
        rows=grid(fixture());parts={};results=[];count=0
        for group,methods in c.GROUPS.items():
            part=[dict(x,provider_group=group,common_standard_noise_sha256='a'*64) for x in rows if x['method'] in methods]
            parts[group]=part;calls=len(part)
            results.append(dict(group=group,logical_rows=group,logical_count=len(part),actual_new_packet_calls=calls,
                initial_ledger=dict(total=count,unresolved=0,cap=7200),final_ledger=dict(total=count+calls,unresolved=0,cap=7200)))
            count+=calls
        return rows,parts,results
    def test_all3600_join_and_common900_noise_identity(self):
        rows,parts,results=self.results();joined,ledger=c.close_groups(results,rows,parts.__getitem__)
        self.assertEqual(len(joined),3600);self.assertEqual(ledger,dict(total=3600,unresolved=0,cap=7200))
    def test_different_common_noise_stops_before_completion(self):
        rows,parts,results=self.results();parts['whole'][0]['common_standard_noise_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'different standard AWGN'):c.close_groups(results,rows,parts.__getitem__)
    def test_unresolved_paid_call_cannot_close(self):
        rows,parts,results=self.results();results[1]['final_ledger']['unresolved']=1
        with self.assertRaisesRegex(ValueError,'ledger continuity'):c.close_groups(results,rows,parts.__getitem__)


if __name__=='__main__':unittest.main(verbosity=2)
