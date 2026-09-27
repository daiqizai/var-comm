import copy
import unittest
from unittest.mock import patch
from tools import publish_c_repeat_initial as p


def registration(group='m6',seed=2026092404):
    cfg=dict(repeats={'additional_training_seeds':list(p.SEEDS)},
             data_seed=2026092305,channel_seed=2026092306,initialization_seed=2026092304)
    r=dict(group=group,seed=seed,parent=None,protocol=cfg,
           parameter_counts={a:1 for a in p.ARMS[group]},
           order_seed=seed+1,channel_seed=seed+2)
    if group=='pure':
        r.update(N=4084,fresh_pure_repeat=True,
                 cache_train={'source':'original F, checked shard hashes'},
                 cache_calibration={'source':'original F, checked shard hashes'})
    else:
        nd=1200 if group=='m6' else 2992
        r['cache_calibration']={'scope':{'ledger':dict(N=4084,NH=68,ND=nd,NA=4016-nd,E=8168)}}
    return r


class RepeatPublicationTests(unittest.TestCase):
    def test_exact_registered_scope_and_engines(self):
        for seed in p.SEEDS:
            for group in p.ARMS:
                p.validate_scope(registration(group,seed),group,seed)
                src,cmd,target=p.scope(group,seed)
                self.assertEqual(cmd[0],'token_efficiency.C_train' if group=='pure' else 'short_prefix.train')
                self.assertEqual('scoped_N4084' in str(src),group=='pure')
                self.assertIn(str(seed),str(target))
        for group,seed in [('m7',p.SEEDS[0]),('m6',2026092304),('pure',0)]:
            with self.assertRaises(ValueError):p.scope(group,seed)

    def test_fresh_pure_rejects_inherited_parent_or_wrong_budget(self):
        for updates in [dict(parent={'checkpoint':'old'}),dict(N=3060),dict(fresh_pure_repeat=False)]:
            r=registration('pure');r.update(updates)
            with self.assertRaises(ValueError):p.validate_scope(r,'pure',p.SEEDS[0])

    def test_hybrid_rejects_scoped_engine_and_wrong_resource(self):
        for field,value in [('N',4084),('fresh_pure_repeat',False)]:
            r=registration();r[field]=value
            with self.assertRaises(ValueError):p.validate_scope(r,'m6',p.SEEDS[0])
        r=registration();r['cache_calibration']['scope']['ledger']['NA']=1792
        with self.assertRaises(ValueError):p.validate_scope(r,'m6',p.SEEDS[0])

    def test_arm_count_and_rng_are_scope(self):
        r=registration('m8');r['parameter_counts']['H8-P']=1
        with self.assertRaises(ValueError):p.validate_scope(r,'m8',p.SEEDS[0])
        r=registration();r['order_seed']+=1
        with self.assertRaises(ValueError):p.validate_scope(r,'m6',p.SEEDS[0])

    def test_complete_boundary_not_partial_or_wrong_arm(self):
        a=['H8-V'];x=dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
            synthetic=False,registration_sha256='r',state=dict(step=20000,last_full=20000,
            updates={'H8-V':20000}),selected={'H8-V':{}})
        p.validate_boundary(x,'r',a)
        y=copy.deepcopy(x);y['state']['last_full']=17500
        with self.assertRaises(ValueError):p.validate_boundary(y,'r',a)
        with self.assertRaises(ValueError):p.validate_boundary(x,'r',['H8-V','H8-P'])
        y=copy.deepcopy(x);y['synthetic']=True
        with self.assertRaises(ValueError):p.validate_boundary(y,'r',a)

    def test_zero_optimizer_exception_is_only_for_zero(self):
        def value(step,filled):
            return dict(registration_sha256='r',state=dict(step=step,last_full=step,
                updates={'P4084':step}),models={'P4084.weight':1},
                optimizers={'P4084':{'state':{0:1} if filled else {}}},
                order={},rng=[],torch_rng=[],cuda_rng=[])
        p.checkpoint_payload(value(0,False),0,'r',['P4084'])
        p.checkpoint_payload(value(20000,True),20000,'r',['P4084'])
        for step,filled in [(0,True),(20000,False)]:
            with self.assertRaises(ValueError):p.checkpoint_payload(value(step,filled),step,'r',['P4084'])
        x=value(20000,True);del x['rng']
        with self.assertRaises(ValueError):p.checkpoint_payload(x,20000,'r',['P4084'])

    def test_repeat_pure_selected_has_zero_parent_updates(self):
        curves=[dict(method='P4084',step=0,utility=2.0),dict(method='P4084',step=17500,utility=1.0),
                dict(method='P4084',step=20000,utility=1.1)]
        x={'P4084':dict(arm_key='P4084',registration_sha256='r',step=17500,total_updates=17500,
                       parent_updates=0,utility=1.0,checkpoint='x',checkpoint_sha256='h')}
        with patch.object(p,'digest',return_value='h'):
            p.validate_selected(x,curves,'r',['P4084'])
            x['P4084']['parent_updates']=10000
            with self.assertRaises(ValueError):p.validate_selected(x,curves,'r',['P4084'])

    def test_stage_command_checked_before_snapshot_even_with_gap_option(self):
        from pathlib import Path
        with patch.object(Path,'exists',return_value=True),patch.object(p,'read',return_value={'command':['wrong']}):
            with self.assertRaisesRegex(ValueError,'exact engine'):
                p.boundary(Path('/x'),['short_prefix.train'],'m6',p.SEEDS[0],'r',True)

    def test_missing_stage_requires_explicit_gap(self):
        from pathlib import Path
        with patch.object(Path,'exists',return_value=False):
            with self.assertRaisesRegex(ValueError,'explicit thermal'):
                p.boundary(Path('/x'),['short_prefix.train'],'m6',p.SEEDS[0],'r',False)

    def test_pure_crc_schema_stays_not_applicable(self):
        row=dict(method='P4084',source_index='0',snr_db='1',seed='4101',image_id='a',
                 header_ok='1',body_crc_ok='not_applicable',mse='1',lpips_alex='2',
                 normalized_latent='3',utility='1.23',U_image='1.2')
        p.validate_pure_rows([row],[{'image_id':'a'}],[1],[4101])
        self.assertEqual(row['body_crc_ok'],'not_applicable')
        row['body_crc_ok']='1'
        with self.assertRaises(ValueError):p.validate_pure_rows([row],[{'image_id':'a'}],[1],[4101])


if __name__=='__main__':
    unittest.main()
