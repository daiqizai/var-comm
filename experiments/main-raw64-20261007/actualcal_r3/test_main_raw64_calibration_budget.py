import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import main_raw64_calibration_budget as b


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.canonical(value), encoding='utf-8')
    return str(path)


def fixture(root, actual_cap=2):
    ledger_module = Path(os.environ.get('MAIN_RAW64_ROOT_LEDGER_MODULE', str(Path(__file__).absolute().parents[1]/'main_raw64_ledger.py')))
    spec = importlib.util.spec_from_file_location('test_original_raw64_ledger', ledger_module)
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    rootreg = root/'root_registration.json'; cfgpath = root/'root_config.json'; ledgerpath = root/'budget.sqlite'
    caps = dict(qualification=1, proxy=1, actual_calibration=actual_cap, development=3, raw_holdout=3, online_PHY=3, optional_H_holdout=3)
    put(cfgpath, dict(registration=str(rootreg), ledger=str(ledgerpath)))
    put(rootreg, dict(status='MAIN_RAW64_CPU_REGISTERED_V1', config_sha256=b.file_sha(cfgpath), phase_caps=caps, source_bindings={str(ledger_module):b.file_sha(ledger_module)}))
    rootsha = b.file_sha(rootreg); ledger = original.Ledger.create(ledgerpath, rootsha, caps)
    for phase in ('qualification', 'proxy'):
        ledger.decode_once(dict(phase=phase, kind='body', phy_key='wire', event_id=phase), {}, lambda:{'actual':True}, {'pid':1})
    before = ledger.quiescent()
    outputs = {}
    for phase in ('qualification', 'proxy'):
        p = root/(phase+'.json')
        put(p, dict(status='MAIN_RAW64_CPU_PHASE_COMPLETE_V1', phase=phase, registration_sha256=rootsha, packet_calls=1, workers_waited=1 if phase=='qualification' else 2))
        outputs[str(p)] = b.file_sha(p)
    identity = dict(pid=33, start_ticks=11, uid=1002, argv=['original_python','-B','original_runner','--registration',str(rootreg),'--run'])
    done = put(root/'normal.json', dict(status='MAIN_RAW64_QUALIFICATION_PROXY_NORMALLY_COMPLETE_V1', registration_sha256=rootsha, automatic_successor_started=False, outputs=outputs, owner_identity=identity, budget=before))
    log = root/'owner.log'; log.write_text('Actual fixture process was waited\n')
    end = put(root/'owner_exit.json', dict(process_waited=True, exit_code=0, identity=identity, log=str(log), log_sha256=b.file_sha(log)))
    reg = dict(status=b.STATUS, phase=b.PHASE, original_ledger_module=str(ledger_module), root_registration=str(rootreg), root_config=str(cfgpath), root_completion=done, root_owner_exit=end, ledger=str(ledgerpath), phase_caps=caps, budget_before=before, stage_call_cap=actual_cap, event_namespace='MAIN_CALIBRATION_V1', plan_sha256='a'*64,
        source_bindings={str(Path(b.__file__).absolute()):b.file_sha(b.__file__),str(ledger_module):b.file_sha(ledger_module),str(Path(b.__file__).absolute().with_name('main_raw64_calibration_checkpoints.py')):b.file_sha(Path(b.__file__).with_name('main_raw64_calibration_checkpoints.py'))}, input_bindings={p:b.file_sha(p) for p in [str(rootreg),str(cfgpath),done,end,str(log),*outputs]})
    rp = put(root/'stage_registration.json',reg)
    return reg, rp, ledger


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.reg, self.path, self.rootledger = fixture(self.root)

    def event(self, index='one', phase='actual_calibration'):
        return dict(event_id='MAIN_CALIBRATION_V1/'+index,phase=phase,kind='header',phy_key='original_header')

    def test_same_database_original_meta_and_immutable_single_authority(self):
        original = b.file_sha(self.reg['original_ledger_module'])
        with self.assertRaisesRegex(ValueError,'not been'): b.CalibrationBudget(self.path)
        gate = b.CalibrationBudget.install(self.path)
        self.assertEqual(gate.ledger.path, self.reg['ledger'])
        self.assertEqual(gate.ledger.registration, self.rootledger.registration)
        self.assertEqual(gate.snapshot(), self.reg['budget_before'])
        with self.assertRaisesRegex(ValueError,'already'): b.CalibrationBudget.install(self.path)
        self.assertEqual(b.file_sha(self.reg['original_ledger_module']), original)
        with self.assertRaisesRegex(ValueError,'Unimplemented'):
            self.rootledger.decode_once(self.event(), {}, lambda:None, {})

    def test_precharge_exact_reuse_never_charges_other_phase(self):
        gate = b.CalibrationBudget.install(self.path); called = []
        def callback():
            called.append(1); snap = gate.snapshot()
            self.assertEqual(snap['phase_charged']['actual_calibration'],1)
            self.assertEqual(snap['unresolved'],1)
            return {'actual_hard_payload':[0,1],'crc':False}
        value = gate.decode_once(self.event(), {'RX':'same'}, callback, {'pid':2})
        self.assertEqual(gate.decode_once(self.event(), {'RX':'same'}, callback, {'pid':2}),value)
        self.assertEqual(len(called),1)
        self.assertEqual(gate.snapshot()['phase_charged']['proxy'],1)
        for phase in ('qualification','proxy','development','raw_holdout','online_PHY','optional_H_holdout'):
            with self.assertRaisesRegex(ValueError,'Only'): gate.decode_once(self.event(phase,phase),{},callback,{})

    def test_concurrent_stage_cap_and_no_refund(self):
        self.reg['stage_call_cap'] = 1; put(Path(self.path),self.reg)
        gate=b.CalibrationBudget.install(self.path); errors=[];barrier=threading.Barrier(2)
        def run(i):
            try:
                barrier.wait();gate.decode_once(self.event(str(i)),{},lambda:{'ok':True},{'pid':i})
            except ValueError as error: errors.append(str(error))
        threads=[threading.Thread(target=run,args=(i,)) for i in range(2)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(len(errors),1)
        self.assertEqual(gate.snapshot()['charged'],3)
        self.assertEqual(gate.snapshot()['phase_charged']['actual_calibration'],1)

    def test_failed_callback_permanently_charged_and_blocks_next(self):
        gate=b.CalibrationBudget.install(self.path)
        def fail():raise RuntimeError('decoder failure')
        with self.assertRaises(RuntimeError):gate.decode_once(self.event(),{},fail,{})
        with self.assertRaisesRegex(ValueError,'retry'):gate.decode_once(self.event(),{},lambda:None,{})
        with self.assertRaisesRegex(ValueError,'Failed root'):gate.decode_once(self.event('next'),{},lambda:None,{})
        self.assertEqual(gate.snapshot()['phase_charged']['actual_calibration'],1)
        self.assertEqual(gate.snapshot()['unresolved'],1)

    def test_missing_normal_exit_or_changed_predecessor_rejected_before_mutation(self):
        before=Path(self.reg['ledger']).read_bytes()
        end=b.read(self.reg['root_owner_exit']);end['exit_code']=1;put(Path(self.reg['root_owner_exit']),end)
        self.reg['input_bindings'][self.reg['root_owner_exit']]=b.file_sha(self.reg['root_owner_exit']);put(Path(self.path),self.reg)
        with self.assertRaisesRegex(ValueError,'wait0'):b.CalibrationBudget.install(self.path)
        self.assertEqual(Path(self.reg['ledger']).read_bytes(),before)
        with self.rootledger.connect() as db:
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='stage_authorities'").fetchone())

    def test_no_second_ledger_or_duplicate_budget(self):
        self.reg['ledger']=str(self.root/'second.sqlite');put(Path(self.path),self.reg)
        with self.assertRaisesRegex(ValueError,'root ledger'):b.CalibrationBudget.install(self.path)
        self.assertFalse((self.root/'second.sqlite').exists())
        self.reg['ledger']=self.rootledger.path;self.reg['stage_call_cap']=3;put(Path(self.path),self.reg)
        with self.assertRaisesRegex(ValueError,'expand'):b.CalibrationBudget.install(self.path)


if __name__=='__main__':unittest.main()
