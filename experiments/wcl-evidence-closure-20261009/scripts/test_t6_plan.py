"""Synthetic metadata contracts only; no real constructors, PHY or models."""
import unittest
from pathlib import Path
import t6_plan as p
import t6_source_audit as a

ROOT=Path(__file__).resolve().parents[3]
SNAP=ROOT/'.research/wcl_evidence_closure_20261009/remote_snapshot'

def original_modules(tag):
    core_path=SNAP/'main_action_space.py';planner_path=SNAP/'source/experiments/prior-aware-grouped-mcs-v1/profiles.py'
    if not core_path.exists():core_path=ROOT/'outputs/CONTENT-REAL-64QAM-20261006/prepared_v1/main_plan_only_runtime/main_action_space.py'
    if not planner_path.exists():planner_path=ROOT/'experiments/prior-aware-grouped-mcs-v1/profiles.py'
    p.h.require(p.h.sha(core_path)==p.CORE_SHA and p.h.sha(planner_path)==p.FIXED['original_planner'],'Frozen finite-rule fixtures changed')
    core=p.h.load(str(core_path),'_t6_test_core_'+tag,{str(core_path):p.h.sha(core_path)})
    original=p.h.load(str(planner_path),'_t6_test_profiles_'+tag,{str(planner_path):p.h.sha(planner_path)})
    return p.configure_core(core),original

class FakeBackend:
    identity={'synthetic':True};qualified=False
    def plan(self,k,n,num_bits_per_symbol):
        row=dict(k=k,n=n,k_ldpc=k,k_filler=0,n_cb=n,z=2,bg='bg1',code_blocks=1,mother_bits=n,
            puncturing_bits=0,shortening_bits=0,repetition_bits=0,decoder_config={'synthetic':True},
            bit_mapping='synthetic',scrambling='synthetic',power_protocol='fixed_constellation_average_Es2')
        row['layout_id']=p.h.digest(row);return row

class ResourceContracts(unittest.TestCase):
    def test_resource_queries_preserve_raw_bits_crc_alignment_and_budget(self):
        q=p.queries();self.assertEqual(len({(x['k'],x['n'],x['num_bits_per_symbol'])for x in q}),len(q))
        self.assertTrue(all(x['k']==12*x['source_token_count']+16 and x['n']%x['num_bits_per_symbol']==0
            and x['n']//x['num_bits_per_symbol']<=1980 for x in q))
        self.assertTrue(any(x['k']==8176 and x['num_bits_per_symbol']==6 for x in q))

    def test_original_finite_generator_retains_whole_permissions_and_fullscale(self):
        core,old=original_modules('whole');enumeration=core.enumerate_actions(old.Planner(FakeBackend()));cat=p.catalogue(enumeration)
        self.assertTrue(any(x['m']==10 and x['K']==0 for x in cat['profiles']))
        self.assertTrue(any(x['K']==0 and'full_budget'in x['allocation_modes']for x in cat['profiles']))
        self.assertTrue(all('PARTIAL_WITH_WHOLE_FALLBACK'in x['family_memberships']for x in cat['profiles']))
        self.assertTrue(all(x['N']==2048 and x['header_symbols']==68 for x in cat['profiles']))
        for rule in enumeration['generation']:
            L=core.SIZES[rule['m']]**2;wanted={0,L//4,L//2,3*L//4}
            if rule['max_legal_K']is not None:wanted.add(rule['max_legal_K'])
            self.assertEqual(rule['K_candidates'],sorted(wanted))

    def test_backend_nonmonotone_rejection_does_not_assume_maximum_is_legal(self):
        core,old=original_modules('nonmonotone')
        class RejectNextScale(FakeBackend):
            def plan(self,k,n,num_bits_per_symbol):
                if k==12*core.PREFIX[6]+16:raise old.UnsupportedConfiguration('Synthetic lifting boundary rejection')
                return super().plan(k,n,num_bits_per_symbol)
        planner=core.MainPlanner(old.Planner(RejectNextScale()))
        self.assertEqual(planner.max_K(5,core.MCS('64QAM','5/6'),'nominal'),35)

    def test_complete_next_scale_is_canonical_whole(self):
        core,_=original_modules('canonical');self.assertEqual(core.canonical(9,256),(10,0))
        with self.assertRaises(ValueError):core.canonical(10,1)

    def test_source_aliases_cannot_make_seen_image_look_unseen(self):
        self.assertEqual(a.source_key('n07836838/ILSVRC2012_val_00008223_n07836838'),a.source_key('/data/n07836838/ILSVRC2012_val_00008223_n07836838.JPEG'))
        self.assertNotEqual(a.source_key('train-00000-of-00294.parquet:000001'),a.source_key('train-00000-of-00294.parquet:000002'))
        with self.assertRaisesRegex(RuntimeError,'duplicate'):a.records(['n/ILSVRC2012_val_00000001_n','/data/n/ILSVRC2012_val_00000001_n.JPEG'])

if __name__=='__main__':unittest.main()
