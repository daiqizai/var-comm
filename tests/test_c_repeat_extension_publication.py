import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import publish_c_repeat_extension as p


class RepeatExtensionPublicationTests(unittest.TestCase):
    def test_scopes_and_checkpoint_engines(self):
        for seed in p.SEEDS:
            for group in p.ARMS:
                src,cmd,base,target=p.scope(group,seed,30000)
                self.assertEqual(cmd[-1],'30000')
                self.assertIn(str(seed),target.name)
                self.assertEqual(cmd[0],'token_efficiency.C_train' if group=='pure' else 'short_prefix.train')
                cp=p.checkpoint_path(src,group,30000)
                self.assertEqual(cp.name,'step_30000_full_calibration.pt' if group=='pure' else 'step_30000.pt')
        for until in (20000,25000,30001,True):
            with self.assertRaises(ValueError):p.scope('m6',p.SEEDS[0],until)
        with self.assertRaises(ValueError):p.scope('m7',p.SEEDS[0],30000)
        with self.assertRaises(ValueError):p.scope('pure',2026092304,30000)

    def previous(self,group='m6'):
        return dict(N=4084,group=group,training_seed=p.SEEDS[0],until=20000,
                    synthetic=False,terminal_artifacts_verified=True,parent=None,parent_updates=0,
                    arms=p.ARMS[group],terminal_checkpoint=dict(step=20000,path='cp',sha256='sha'),
                    official_decision=dict(step=20000,extend=True,until=30000))

    def test_prior_must_be_same_fresh_seed_and_authorized(self):
        a=self.previous('pure')
        with patch.object(p,'digest',return_value='sha'):
            self.assertEqual(p.validate_previous(a,'pure',p.SEEDS[0],30000)['step'],20000)
            for k,v in [('N',3060),('training_seed',p.SEEDS[1]),('parent_updates',10000),
                        ('synthetic',True),('terminal_artifacts_verified',False),('until',10000)]:
                bad=copy.deepcopy(a);bad[k]=v
                with self.assertRaises(ValueError):p.validate_previous(bad,'pure',p.SEEDS[0],30000)
            bad=copy.deepcopy(a);bad['official_decision']['extend']=False
            with self.assertRaises(ValueError):p.validate_previous(bad,'pure',p.SEEDS[0],30000)

    def test_history_requires_every_prior_publication(self):
        with tempfile.TemporaryDirectory() as d, patch.object(p.initial,'RESULTS',Path(d)):
            # No fallback to another seed's completed first matrix.
            with self.assertRaises(FileNotFoundError):p.history('m6',p.SEEDS[0],40000,'reg')

    def test_boundary_rejects_incomplete_or_wrong_arms(self):
        a=dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',synthetic=False,
               registration_sha256='reg',state=dict(step=30000,last_full=30000,updates={'H8-V':30000}),
               selected={'H8-V':{}})
        p.validate_boundary(a,'reg',['H8-V'],30000)
        for field,value in [('last_full',27500),('updates',{'H8-V':30000,'H8-P':30000})]:
            bad=copy.deepcopy(a);bad['state'][field]=value
            with self.assertRaises(ValueError):p.validate_boundary(bad,'reg',['H8-V'],30000)

    def test_success_uses_immutable_snapshot_not_live_completion(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p,'OUT',Path(d)):
            root=Path(d);stage=root/'delivery_chain_v1/stages/C_N4084_m8_seed2026092404_until30000.json'
            stage.parent.mkdir(parents=True)
            snap=root/'snapshot.json';live=root/'completion.json'
            done=dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',synthetic=False,
                      registration_sha256='reg',state=dict(step=30000,last_full=30000,updates={'H8-V':30000}),
                      selected={'H8-V':{}})
            snap.write_text(json.dumps(done));live.write_text('{"state":{"step":40000}}')
            command=p.scope('m8',p.SEEDS[0],30000)[1]
            data=dict(command=list(command),returncode=0,snapshot=str(snap),
                      completion=str(live),completion_sha256=p.digest(snap))
            stage.write_text(json.dumps(data))
            got,evidence,_=p.boundary(root,command,'m8',p.SEEDS[0],'reg',False,30000)
            self.assertEqual(got,done);self.assertTrue(evidence['complete_stage'])
            data['command'][-1]='20000';stage.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                p.boundary(root,command,'m8',p.SEEDS[0],'reg',True,30000)

    def test_gap_never_implicit(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p,'OUT',Path(d)):
            with self.assertRaises(ValueError):
                p.boundary(Path(d),[],'m6',p.SEEDS[0],'reg',False,30000)

    def test_old_rounds_reference_exact_owner_and_new_rounds_retained(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p.initial,'RESULTS',Path(d)):
            root=Path(d);base=p.scope('m6',p.SEEDS[0],30000)[2]
            ext=p.scope('m6',p.SEEDS[0],30000)[3]
            files={}
            for step,owner in [(20000,base),(22500,ext),(30000,ext),(32500,None)]:
                n=f'calibration/full_{step:05d}.csv';actual=root/f'actual_{step}.csv'
                actual.write_text(str(step));files[n]=actual
                if owner:
                    q=owner/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(str(step))
            kept=p.retained_sources(files,'m6',p.SEEDS[0],40000)
            self.assertEqual(list(kept),['calibration/full_32500.csv'])
            (ext/'calibration/full_30000.csv').write_text('changed')
            with self.assertRaises(ValueError):p.retained_sources(files,'m6',p.SEEDS[0],40000)

    def test_prepare_before20k_reports_missing_parent_without_boundary_claim(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p.initial,'RESULTS',Path(d)),\
             patch.object(p,'OUT',Path(d)),patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),\
             patch.object(p.initial,'audit_inputs',return_value=({},'reg',{})),\
             patch.object(p.initial,'calibration',return_value=([],[],{0:[]},{},30000)),\
             patch.object(p.initial,'validate_selected'),patch.object(p.initial,'cpu_checkpoint',return_value={}),\
             patch.object(p,'read',return_value={'checkpoint':'cp','step':0}):
            a=p.prepare('m6',p.SEEDS[0],30000)
            self.assertFalse(a['boundary_published'])
            self.assertFalse(a['previous_boundary_verified'])
            self.assertIsNone(a['parent_checkpoint'])
            self.assertEqual(len(a['missing_previous_publications']),1)
            self.assertEqual(a['status'],'C_REPEAT_EXTENSION_CPU_PREPARATION_ONLY')



    def test_original_hybrid_completion_without_synthetic_field(self):
        for arms in (['H6-V','H6-P'],['H8-V'],['P4084']):
            x=dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
                   registration_sha256='reg',state=dict(step=30000,last_full=30000,
                   updates={a:30000 for a in arms}),selected={a:{} for a in arms})
            if arms==['P4084']:
                with self.assertRaises(ValueError):p.validate_boundary(x,'reg',arms,30000)
            else:
                p.validate_boundary(x,'reg',arms,30000)
            for value in (True,None,0,'false'):
                x['synthetic']=value
                with self.assertRaises(ValueError):p.validate_boundary(x,'reg',arms,30000)
            x['synthetic']=False
            p.validate_boundary(x,'reg',arms,30000)


if __name__=='__main__':unittest.main()
