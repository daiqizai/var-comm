"""Synthetic CPU regressions for scoped publication gating; no GPU quality claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import publish_c_n3060_extension as p


class N3060ExtensionPublicationTests(unittest.TestCase):
    def test_only_registered_10k_boundaries(self):
        for n in (20000, 22500, 30001, -1, True):
            with self.assertRaises(ValueError):
                p.target_for(n)
        self.assertEqual(p.target_for(40000).name, 'N3060_m6_seed2026092304_until40000')

    def test_scoped_n3060_only(self):
        reg=dict(N=3060,group='m6',seed=p.SEED,parent=None,fresh_pure_repeat=False,
                 cache_calibration=dict(scope=dict(ledger=dict(N=3060,NH=68,ND=1200,NA=1792,E=6120))))
        p.validate_scope(reg)
        for key,value in [('N',4084),('group','m7'),('seed',2026092404),('parent',{})]:
            bad=copy.deepcopy(reg);bad[key]=value
            with self.assertRaises(ValueError):
                p.validate_scope(bad)

    def boundary(self,step=30000):
        return dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
                    synthetic=False,registration_sha256='reg',
                    state=dict(step=step,last_full=step,updates={a:step for a in p.ARMS}),
                    selected={a:{} for a in p.ARMS})

    def test_full_pair_required(self):
        p.validate_boundary(self.boundary(),'reg',30000)
        for bad in [self.boundary(20000),self.boundary()]:
            if bad['state']['step']==30000:
                bad['state']['updates']['H6-P']=29999
            with self.assertRaises(ValueError):
                p.validate_boundary(bad,'reg',30000)

    def test_stage_snapshot_survives_mutable_completion(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(p,'OUT',Path(tmp)):
            src=Path(tmp)/'run';src.mkdir()
            (src/'completion.json').write_text(json.dumps(self.boundary(40000)))
            snap=Path(tmp)/'snapshot.json';snap.write_text(json.dumps(self.boundary()))
            stage=Path(tmp)/'delivery_chain_v1/stages'/f'C_N3060_m6_seed{p.SEED}_until30000.json'
            stage.parent.mkdir(parents=True)
            value=dict(command=['token_efficiency.C_train','--N','3060','--group','m6',
                                '--seed',str(p.SEED),'--until','30000'],
                       returncode=0,snapshot=str(snap),completion_sha256=p.digest(snap))
            stage.write_text(json.dumps(value))
            done,evidence,_=p.capture_boundary(src,'reg',False,30000)
            self.assertEqual(done['state']['step'],30000)
            self.assertTrue(evidence['complete_stage'])
            value['returncode']=-9;stage.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                p.capture_boundary(src,'reg',True,30000)

    def test_missing_stage_needs_explicit_gap_and_exact_terminal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(p,'OUT',Path(tmp)):
            src=Path(tmp)/'run';src.mkdir()
            (src/'completion.json').write_text(json.dumps(self.boundary(20000)))
            for allow in (False,True):
                with self.assertRaises(ValueError):
                    p.capture_boundary(src,'reg',allow,30000)

    def test_delta_only_and_historical_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);initial=root/'initial';files={}
            for step in range(0,30001,2500):
                for suffix in ('.csv','.json'):
                    name=f'calibration/full_{step:05d}{suffix}'
                    source=root/'live'/name;source.parent.mkdir(parents=True,exist_ok=True)
                    source.write_text(str(step)+suffix);files[name]=source
                    if step<=20000:
                        old=initial/name;old.parent.mkdir(parents=True,exist_ok=True)
                        old.write_text(source.read_text())
            with patch.object(p,'INITIAL',initial):
                retained=p.publication_sources(files,30000)
                self.assertEqual(len(retained),8)
                self.assertTrue(all(int(Path(k).stem.split('_')[1])>20000 for k in retained))
                files['calibration/full_00000.csv'].write_text('tampered')
                with self.assertRaises(ValueError):
                    p.publication_sources(files,30000)

    def test_later_boundary_requires_previous_publication(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(p,'INITIAL',Path(tmp)/'initial'), patch.object(p,'RESULTS',Path(tmp)):
            with self.assertRaises((ValueError,FileNotFoundError)):
                p.history(40000,'reg')


if __name__=='__main__':
    unittest.main()
