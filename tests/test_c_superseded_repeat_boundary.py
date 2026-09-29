import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import audit_c_superseded_repeat_boundary as h
from tools import publish_c_repeat_extension as p


class SupersededBoundaryTests(unittest.TestCase):
    def test_superseded_requires_formal_extension_and_same_registration(self):
        cur = dict(registration_sha256='reg', state=dict(step=40000,last_full=40000))
        d = dict(step=30000,extend=True,until=40000,development_used=False)
        h.validate_superseded(cur,'reg',30000,d)
        for field,value in [('extend',False),('until',50000),('development_used',True)]:
            bad=copy.deepcopy(d);bad[field]=value
            with self.assertRaises(ValueError):h.validate_superseded(cur,'reg',30000,bad)
        for step in (20000,30000):
            bad=copy.deepcopy(cur);bad['state']['step']=step
            with self.assertRaises(ValueError):h.validate_superseded(bad,'reg',30000,d)
        with self.assertRaises(ValueError):h.validate_superseded(cur,'wrong',30000,d)

    def test_terminal_log_rejects_later_output_errors_and_wrong_metrics(self):
        s={'H6-V':.02,'H6-P':.03}
        line="full calibration 30000 "+repr(s)
        h.terminal_log_summary('train m6 30000\n'+line+'\n',30000,s)
        for log in (line+'\ntrain m6 30100', 'Traceback\n'+line,
                    line.replace('30000','20000'),line.replace('0.03','0.04')):
            with self.assertRaises(ValueError):h.terminal_log_summary(log,30000,s)

    def test_old_selected_checkpoint_is_preserved(self):
        state=dict(step=30000,selection={'H6-V':dict(step=27500,utility=.02),
                                        'H6-P':dict(step=30000,utility=.021)})
        with patch.object(h,'digest',return_value='sha'):
            got=h.selected_from_checkpoint(Path('/original'),state,['H6-V','H6-P'],'reg')
            self.assertEqual(got['H6-V']['step'],27500)
            self.assertTrue(got['H6-V']['checkpoint'].endswith('step_27500.pt'))
            self.assertEqual(got['H6-V']['total_updates'],27500)
            state['selection']['H6-V']['step']=32500
            with self.assertRaises(ValueError):
                h.selected_from_checkpoint(Path('/original'),state,['H6-V','H6-P'],'reg')

    def test_superseded_audit_is_opt_in_and_does_not_replace_valid_stage(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p,'OUT',Path(d)),\
             patch.object(h,'audit',return_value=('derived',{},{})) as audit:
            src=Path(d);cmd=p.scope('m6',p.SEEDS[1],30000)[1]
            with self.assertRaises(ValueError):
                p.boundary(src,cmd,'m6',p.SEEDS[1],'reg',False,30000)
            audit.assert_not_called()
            self.assertEqual(p.boundary(src,cmd,'m6',p.SEEDS[1],'reg',False,30000,True)[0],'derived')
            stage=src/'delivery_chain_v1/stages/C_N4084_m6_seed2026092504_until30000.json'
            stage.parent.mkdir(parents=True);stage.write_text('{}')
            with self.assertRaises(KeyError):
                p.boundary(src,cmd,'m6',p.SEEDS[1],'reg',False,30000,True)
            self.assertEqual(audit.call_count,1)

    def test_pure_engine_is_rejected_before_artifact_read(self):
        with self.assertRaises(ValueError):
            h.audit(Path('/not-read'),[],'pure',p.SEEDS[1],'reg',30000)


if __name__ == '__main__':
    unittest.main()
