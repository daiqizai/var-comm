import copy
import tempfile
import unittest
from pathlib import Path
from tools.audit_c_terminal_boundary import validate_artifacts, validate_thermal_sequence
from tools.publish_c_initial_milestone import digest

class TerminalBoundaryEvidenceTests(unittest.TestCase):
    def sequence(self):
        outer=dict(command=['python','-u','-m','token_efficiency.C_followups'],time=1,process={'pid':1})
        launch=dict(time=2,process={'pid':2})
        hot=dict(time=3,temperature=84,software_thermal_slowdown=True,hardware_thermal_slowdown=False)
        stop=dict(reason='thermal',process=launch['process'],hardware=hot)
        outer_stop=dict(reason='thermal',process=outer['process'],hardware=hot)
        resumed=dict(command=outer['command'],time=5)
        return [launch,stop,outer,outer_stop,resumed,4,6]

    def test_only_ordered_observed_thermal_stop_accepted(self):
        args=self.sequence();validate_thermal_sequence(*args)
        for position,field,value in [(1,'reason','requested_stop'),(3,'process',{'pid':9}),
                                    (4,'time',3)]:
            bad=copy.deepcopy(args);bad[position][field]=value
            with self.assertRaises(ValueError):validate_thermal_sequence(*bad)
        bad=copy.deepcopy(args);bad[1]['hardware']['software_thermal_slowdown']=False
        with self.assertRaises(ValueError):validate_thermal_sequence(*bad)

    def test_propagated_outer_thermal_stop_requires_ordered_evidence(self):
        args=self.sequence()
        args[1]['reason']='requested_stop'
        args[1]['hardware']=dict(args[1]['hardware'],time=3.2)
        validate_thermal_sequence(*args)
        for position,field,value in [(3,'reason','requested_stop'),(3,'process',{'pid':9})]:
            bad=copy.deepcopy(args);bad[position][field]=value
            with self.assertRaises(ValueError):validate_thermal_sequence(*bad)
        for stamp in [3.2,3.3]:
            bad=copy.deepcopy(args);bad[3]['hardware']['time']=stamp
            with self.assertRaises(ValueError):validate_thermal_sequence(*bad)
        bad=copy.deepcopy(args);bad[3]['hardware']['software_thermal_slowdown']=False
        with self.assertRaises(ValueError):validate_thermal_sequence(*bad)

    def test_no_recovery_from_partial_or_extending_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'checkpoint';p.write_bytes(b'engineering fixture only')
            state=dict(step=30000,last_full=30000,updates={'H8-V':30000})
            done=dict(status='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
                      registration_sha256='r',state=state,selected={'H8-V':{}})
            status=dict(status='MILESTONE_COMPLETE',state=state)
            latest=dict(step=30000,reason='full_calibration',path=str(p),sha256=digest(p))
            decision=dict(step=30000,until=30000,extend=False,development_used=False)
            args=[done,status,latest,decision,30000,['H8-V'],'r']
            validate_artifacts(*args)
            for pos,field,value in [(0,'registration_sha256','bad'),(1,'status','TRAINING'),
                                    (2,'reason','before_full_calibration'),(3,'extend',True)]:
                bad=copy.deepcopy(args);bad[pos][field]=value
                with self.assertRaises(ValueError):validate_artifacts(*bad)
            bad=copy.deepcopy(args);bad[0]['state']['last_full']=27500
            with self.assertRaises(ValueError):validate_artifacts(*bad)
            p.write_bytes(b'corruption')
            with self.assertRaises(ValueError):validate_artifacts(*args)
