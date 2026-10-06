"""Open only actual_calibration in the existing raw64 study ledger.

No database creation, cap changes, refunds or automatic stage registration.
The registered owner separately controls process admission and CPU locks.
This adapter keeps the original event schema, counters and root registration.
"""
import importlib.util
import json
from pathlib import Path
import time

from main_raw64_calibration_checkpoints import canonical, digest, file_sha, require, verify_files

PHASE = 'actual_calibration'
STATUS = 'MAIN_RAW64_ACTUAL_CALIBRATION_BUDGET_REGISTERED_V1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inspect_registration(path):
    path = str(Path(path).absolute()); reg = read(path)
    require(reg['status'] == STATUS and reg['phase'] == PHASE, 'Separate actual_calibration authority required')
    pins = dict(reg['source_bindings'])
    for p, h in reg['input_bindings'].items():
        require(p not in pins or pins[p] == h, 'Conflicting stage pin')
        pins[p] = h
    verify_files(pins)
    require(pins.get(str(Path(__file__).absolute())) == file_sha(__file__), 'Unbound phase gate source')
    support = str(Path(__file__).absolute().with_name('main_raw64_calibration_checkpoints.py'))
    require(pins.get(support) == file_sha(support), 'Unbound phase gate support module')
    required = ['original_ledger_module', 'root_registration', 'root_config', 'root_completion', 'root_owner_exit']
    require(all(pins.get(reg[k]) == file_sha(reg[k]) for k in required), 'Original ledger/normal controls must be directly bound')
    root = read(reg['root_registration']); cfg = read(reg['root_config']); done = read(reg['root_completion']); end = read(reg['root_owner_exit'])
    rootsha = file_sha(reg['root_registration'])
    require(root['status'] == 'MAIN_RAW64_CPU_REGISTERED_V1' and root['config_sha256'] == file_sha(reg['root_config']) and cfg['registration'] == reg['root_registration'] and cfg['ledger'] == reg['ledger'] and root['phase_caps'] == reg['phase_caps'], 'Original immutable root ledger registration differs')
    require(root['source_bindings'].get(reg['original_ledger_module']) == file_sha(reg['original_ledger_module']), 'Phase gate must use original registered ledger implementation')
    require(done['status'] == 'MAIN_RAW64_QUALIFICATION_PROXY_NORMALLY_COMPLETE_V1' and done['registration_sha256'] == rootsha and done['automatic_successor_started'] is False, 'Actual normal initial CPU sequence required')
    require(end['process_waited'] is True and end['exit_code'] == 0 and pins.get(end['log']) == end['log_sha256'] == file_sha(end['log']), 'Original owner wait0 and closed log must be bound')
    require(all(end['identity'][k] == done['owner_identity'][k] for k in ('pid', 'start_ticks', 'uid', 'argv')), 'Normal completion and waited process identity differ')
    require(end['identity']['argv'][-3:] == ['--registration', reg['root_registration'], '--run'], 'Wrong original owner command')
    require(done['outputs'] and all(pins.get(p) == h == file_sha(p) for p, h in done['outputs'].items()), 'Both original normal phase completions must be bound')
    phases = [read(p) for p in done['outputs']]
    require(len(phases) == 2 and {p['phase'] for p in phases} == {'qualification', 'proxy'}, 'Both original phases completed exactly once')
    for phase in phases:
        require(phase['status'] == 'MAIN_RAW64_CPU_PHASE_COMPLETE_V1' and phase['registration_sha256'] == rootsha and phase['packet_calls'] == root['phase_caps'][phase['phase']] and phase['workers_waited'] == (1 if phase['phase'] == 'qualification' else 2), 'Original phase grid/wait count differs')
    before = done['budget']; caps = reg['phase_caps']
    require(before['registration_sha256'] == rootsha and before['caps'] == caps and before['unresolved'] == 0 and before['phase_charged'] == {p: caps[p] if p in ('qualification', 'proxy') else 0 for p in caps}, 'Initial stages must close before any future stage charge')
    require(type(reg['stage_call_cap']) is int and 0 <= reg['stage_call_cap'] <= caps[PHASE] and reg['budget_before'] == before, 'Actual stage cannot expand its original reserved allowance')
    require(isinstance(reg['event_namespace'], str) and reg['event_namespace'] and isinstance(reg['plan_sha256'], str) and len(reg['plan_sha256']) == 64, 'Frozen finite stage plan and event namespace required')
    spec = importlib.util.spec_from_file_location('raw64_root_ledger_for_actualcal', reg['original_ledger_module'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ledger = module.Ledger(reg['ledger'], rootsha, caps)
    return reg, file_sha(path), ledger


class CalibrationBudget:
    def __init__(self, registration_path):
        self.reg, self.stage_sha, self.ledger = inspect_registration(registration_path)
        self.binding = canonical(dict(stage_registration_sha256=self.stage_sha, root_registration_sha256=self.ledger.registration, phase=PHASE, cap=self.reg['stage_call_cap'], plan_sha256=self.reg['plan_sha256'], event_namespace=self.reg['event_namespace']))
        with self.ledger.connect() as db:
            require(db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stage_authorities'").fetchone() is not None, 'Stage authority has not been explicitly installed')
            self._authority(db)

    @classmethod
    def install(cls, registration_path):
        """Explicit registration operation, after normal q/proxy and no charge."""
        reg, stage_sha, ledger = inspect_registration(registration_path)
        require(ledger.quiescent() == reg['budget_before'], 'Current root budget differs from initial normal closure')
        binding = canonical(dict(stage_registration_sha256=stage_sha, root_registration_sha256=ledger.registration, phase=PHASE, cap=reg['stage_call_cap'], plan_sha256=reg['plan_sha256'], event_namespace=reg['event_namespace']))
        with ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            require(not db.execute("SELECT 1 FROM events WHERE status!='COMPLETE' LIMIT 1").fetchone(), 'Failed or unresolved root event prevents stage registration')
            counts = dict(db.execute('SELECT phase,charged FROM phase_counts'))
            require(counts == reg['budget_before']['phase_charged'], 'Root counts changed before authority seal')
            db.execute('CREATE TABLE IF NOT EXISTS stage_authorities (phase TEXT PRIMARY KEY,binding TEXT NOT NULL)')
            require(db.execute('SELECT 1 FROM stage_authorities WHERE phase=?', (PHASE,)).fetchone() is None, 'Phase already has immutable authority; no re-registration')
            db.execute('INSERT INTO stage_authorities VALUES (?,?)', (PHASE, binding))
        return cls(registration_path)

    def _authority(self, db):
        row = db.execute('SELECT binding FROM stage_authorities WHERE phase=?', (PHASE,)).fetchone()
        require(row is not None and row[0] == self.binding, 'Changed or foreign phase authority')

    def decode_once(self, event, request, decode, worker):
        require(event['phase'] == PHASE and event['kind'] in ('header', 'body') and event['event_id'].startswith(self.reg['event_namespace']+'/'), 'Only the registered actual_calibration namespace is open')
        body = canonical(request); request_sha = digest(request)
        with self.ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE'); self._authority(db)
            prior = db.execute('SELECT * FROM events WHERE event_id=?', (event['event_id'],)).fetchone()
            if prior is not None:
                require(prior['phase'] == PHASE and prior['kind'] == event['kind'] and prior['phy_key'] == event['phy_key'] and prior['request'] == body and prior['request_sha'] == request_sha, 'Changed event cannot reuse a charge')
                require(prior['status'] == 'COMPLETE', 'Failed or unresolved event cannot retry')
                value = json.loads(prior['result']); require(digest(value) == prior['result_sha'], 'Stored result hash changed'); return value
            require(not db.execute("SELECT 1 FROM events WHERE status='FAILED' LIMIT 1").fetchone(), 'Failed root ledger blocks decoding')
            require(db.execute('SELECT charged FROM phase_counts WHERE phase=?', (PHASE,)).fetchone()[0] < self.reg['stage_call_cap'], 'Stage budget exhausted; no borrowing')
            require(db.execute("SELECT count(*) FROM events WHERE status='RESERVED'").fetchone()[0] < 2, 'At most two active packet callbacks')
            db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (event['event_id'], PHASE, event['kind'], event['phy_key'], body, request_sha, canonical(worker), 'RESERVED', None, None, None, time.time()))
            db.execute('UPDATE phase_counts SET charged=charged+1 WHERE phase=?', (PHASE,))
        try:
            value = decode(); encoded = canonical(value); result_sha = digest(value)
        except BaseException as error:
            with self.ledger.connect() as db:
                db.execute("UPDATE events SET status='FAILED',error=? WHERE event_id=? AND status='RESERVED'", (type(error).__name__+': '+str(error), event['event_id']))
            raise
        with self.ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE'); self._authority(db)
            require(db.execute("UPDATE events SET status='COMPLETE',result=?,result_sha=? WHERE event_id=? AND status='RESERVED'", (encoded, result_sha, event['event_id'])).rowcount == 1, 'Prepaid event state changed')
        return value

    def snapshot(self):
        value = self.ledger.snapshot()
        require(value['phase_charged'][PHASE] <= self.reg['stage_call_cap'], 'Actual stage charge exceeds authority')
        for phase, count in self.reg['budget_before']['phase_charged'].items():
            if phase != PHASE:
                require(value['phase_charged'][phase] == count, 'Another reserved phase changed during actual calibration')
        return value
