import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import publish_c_n3060_initial as p


class N3060InitialPublicationTests(unittest.TestCase):
    def reg(self):
        return dict(N=3060,group='m6',seed=2026092304,parent=None,
                    fresh_pure_repeat=False,cache_calibration={'scope':{'ledger':
                    dict(N=3060,NH=68,ND=1200,NA=1792,E=6120)}})

    def done(self):
        return dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
                    synthetic=False,registration_sha256='sha',
                    state=dict(step=20000,last_full=20000,
                               updates={'H6-V':20000,'H6-P':20000}),
                    selected={'H6-V':{},'H6-P':{}})

    def test_scoped_resource_and_parent_identity(self):
        p.validate_scope(self.reg())
        for change in [dict(N=4084),dict(group='m7'),dict(seed=2026092404),
                       dict(parent={'checkpoint':'old'}),dict(fresh_pure_repeat=True)]:
            r=self.reg();r.update(change)
            with self.assertRaises(ValueError):
                p.validate_scope(r)
        r=self.reg();r['cache_calibration']['scope']['ledger']['NA']=2816
        with self.assertRaises(ValueError):
            p.validate_scope(r)

    def test_full_calibration_and_paired_updates_required(self):
        p.validate_boundary(self.done(),'sha')
        for key,value in [('last_full',17500),('step',19999),
                          ('updates',{'H6-V':20000,'H6-P':19999})]:
            d=self.done();d['state'][key]=value
            with self.assertRaises(ValueError):
                p.validate_boundary(d,'sha')
        d=self.done();d['synthetic']=True
        with self.assertRaises(ValueError):
            p.validate_boundary(d,'sha')
        with self.assertRaises(ValueError):
            p.validate_boundary(self.done(),'wrong')

    def test_missing_scheduler_receipt_is_never_implicitly_accepted(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'OUT',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'explicit thermal'):
                p.capture_boundary(Path(tmp),'sha',False)

    def test_existing_stage_cannot_be_replaced_by_gap_option(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'OUT',Path(tmp)):
            stage=Path(tmp)/'delivery_chain_v1/stages'/f'{p.KEY}.json'
            stage.parent.mkdir(parents=True)
            p.write(stage,{'command':['short_prefix.train','--group','m6']})
            with self.assertRaisesRegex(ValueError,'scoped engine'):
                p.capture_boundary(Path(tmp),'sha',True)

    def test_initial_boundary_is_not_finalization_and_paired_extension_is_original(self):
        p.validate_boundary(self.done(),'sha')
        curves=[dict(step=s,method=a,utility=v)
                for a,values in [('H6-V',[1.,.99,.98]),('H6-P',[1.,1.01,1.02])]
                for s,v in zip([15000,17500,20000],values)]
        dec=dict(rule=dict(interval=2500,extend_updates=10000,
                          minimum_relative_improvement=.002,both_intervals_required=True),
                 step=20000,until=30000,extend=True,development_used=False,
                 paired_arms_extend_together=True,
                 values={a:[r['utility'] for r in curves if r['method']==a] for a in p.ARMS})
        dec['arms']={a:dict(extend=a=='H6-V',
            relative_improvements=[(x-y)/x for x,y in zip(v,v[1:])]) for a,v in dec['values'].items()}
        self.assertTrue(p.validate_decision(dec,20000,curves,p.ARMS))
        dec['development_used']=True
        with self.assertRaises(ValueError):
            p.validate_decision(dec,20000,curves,p.ARMS)


if __name__=='__main__':
    unittest.main()

