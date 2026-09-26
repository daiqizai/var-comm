import copy
import json
import tempfile
import unittest
from pathlib import Path
from tools.publish_c_extension import (
    arms_for, validate_decision, completed_snapshot, validate_selection, verify_index,
)
from tools.publish_c_initial_milestone import digest


class CExtensionPublicationTests(unittest.TestCase):
    def curves(self):
        return [dict(step=s,method=a,utility=u) for a,values in
                [('H6-V',[1.,.99,.98]),('H6-P',[1.,1.01,1.00])]
                for s,u in zip((25000,27500,30000),values)]

    def decision(self):
        values={'H6-V':[1.,.99,.98],'H6-P':[1.,1.01,1.]}
        arms={}
        for a,v in values.items():
            gains=[(x-y)/x for x,y in zip(v,v[1:])]
            arms[a]=dict(extend=all(x>=.002 for x in gains),relative_improvements=gains)
        return dict(rule=dict(interval=2500,extend_updates=10000,minimum_relative_improvement=.002,
                              both_intervals_required=True),step=30000,arms=arms,values=values,
                    extend=True,until=40000,paired_arms_extend_together=True,development_used=False)

    def test_pair_extends_on_one_arm_both_intervals(self):
        self.assertTrue(validate_decision(self.decision(),30000,self.curves(),arms_for('m6')))
        for field,value in [('extend',False),('until',30000),('development_used',True),
                            ('paired_arms_extend_together',False)]:
            d=self.decision();d[field]=value
            with self.assertRaises(ValueError):
                validate_decision(d,30000,self.curves(),arms_for('m6'))

    def test_negative_interval_preserved_and_insufficient_alone(self):
        d=self.decision();d['arms']['H6-P']['extend']=True
        with self.assertRaises(ValueError):
            validate_decision(d,30000,self.curves(),arms_for('m6'))
        d=self.decision();d['values']['H6-V'][1]=.98
        with self.assertRaises(ValueError):
            validate_decision(d,30000,self.curves(),arms_for('m6'))

    def test_missing_checkpoint_or_wrong_rule_rejected(self):
        with self.assertRaises(ValueError):
            validate_decision(self.decision(),30000,self.curves()[1:],arms_for('m6'))
        d=self.decision();d['rule']['minimum_relative_improvement']=.001
        with self.assertRaises(ValueError):
            validate_decision(d,30000,self.curves(),arms_for('m6'))

    def test_real_arm_cardinality(self):
        self.assertEqual(arms_for('m8'),['H8-V'])
        self.assertEqual(arms_for('pure'),['P4084'])
        with self.assertRaises(ValueError):arms_for('m10')

    def test_stage_uses_immutable_snapshot_not_mutable_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);snap=root/'snapshot.json';live=root/'completion.json'
            done=dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',registration_sha256='r',
                      state=dict(step=30000,last_full=30000,updates={'H8-V':30000}),
                      selected={'H8-V':{}})
            snap.write_text(json.dumps(done));live.write_text('next extension completed')
            stage=dict(returncode=0,snapshot=str(snap),completion=str(live),completion_sha256=digest(snap))
            self.assertEqual(completed_snapshot(stage,30000,['H8-V'],'r'),done)
            for field,value in [('last_full',27500),('updates',{'H8-V':29999}),('step',40000)]:
                bad=copy.deepcopy(done);bad['state'][field]=value;snap.write_text(json.dumps(bad))
                stage['completion_sha256']=digest(snap)
                with self.assertRaises(ValueError):completed_snapshot(stage,30000,['H8-V'],'r')

    def test_selected_may_remain_an_older_best(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'checkpoint.pt';p.write_bytes(b'CPU test fixture, not weights')
            s={'P4084':dict(arm_key='P4084',registration_sha256='r',step=20000,total_updates=20000,
                           parent_updates=10000,utility=.5,checkpoint=str(p),checkpoint_sha256=digest(p))}
            curves=[dict(method='P4084',step=20000,utility=.5),dict(method='P4084',step=30000,utility=.6)]
            validate_selection(s,curves,['P4084'],'r','pure')
            bad=copy.deepcopy(s);bad['P4084']['step']=30000;bad['P4084']['total_updates']=30000
            with self.assertRaises(ValueError):validate_selection(bad,curves,['P4084'],'r','pure')
            bad=copy.deepcopy(s);bad['P4084']['parent_updates']=0
            with self.assertRaises(ValueError):validate_selection(bad,curves,['P4084'],'r','pure')

    def test_index_rejects_corruption_and_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder=root/'publication';folder.mkdir();p=folder/'data.csv';p.write_bytes(b'x')
            index=dict(files={'data.csv':dict(sha256=digest(p),bytes=1)})
            (folder/'index.json').write_text(json.dumps(index));verify_index(folder)
            p.write_bytes(b'y')
            with self.assertRaises(ValueError):verify_index(folder)
            other=root/'outside';other.write_bytes(b'z')
            (folder/'index.json').write_text(json.dumps(dict(files={'../outside':dict(sha256=digest(other),bytes=1)})))
            with self.assertRaises(ValueError):verify_index(folder)
