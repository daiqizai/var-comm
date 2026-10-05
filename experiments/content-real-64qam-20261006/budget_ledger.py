"""Atomic packet precharges with independent H/C quotas and preserved failures.

Only a header/body decoder callback is charged here. Live reservations belonging
to another exact registered process may run concurrently; orphaned or failed
reservations block new physical work. No refund or implicit recovery exists.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

TOTAL_CAPS = {'H': 200000, 'C': 100000}


class BudgetError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] in ('Z', 'X'):
        raise ProcessLookupError('Process has exited')
    return dict(pid=int(pid), start_ticks=int(fields[19]), uid=proc.stat().st_uid,
                argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))


def same_identity(a, b):
    return (all(int(a[k]) == int(b[k]) for k in ('pid', 'start_ticks', 'uid')) and a['argv'] == b['argv'])


class BudgetLedger:
    def __init__(self, path, registration_sha256, branch, limits, worker_identity=None,
                 identity_reader=process_identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.binding, self.branch, self.limits = str(registration_sha256), str(branch), dict(limits)
        if branch not in TOTAL_CAPS or not self.limits or any(type(v) is not int or v < 0 for v in self.limits.values()):
            raise BudgetError('Invalid registered branch or phase limits')
        if 'development' not in self.limits or self.limits['development'] <= 0:
            raise BudgetError('A positive non-transferable development reserve is required')
        if sum(self.limits.values()) > TOTAL_CAPS[branch]:
            raise BudgetError('Phase quotas exceed independent branch cap')
        self.identity_reader = identity_reader
        self.worker = dict(worker_identity or identity_reader(os.getpid()))
        for key in ('pid', 'start_ticks', 'uid', 'argv'):
            if key not in self.worker:
                raise BudgetError('Incomplete worker identity')
        if not same_identity(self.worker, identity_reader(self.worker['pid'])):
            raise BudgetError('Worker identity is not current')
        with self.transaction() as c:
            c.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS configurations (phy_key TEXT PRIMARY KEY,definition TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS counters (phase TEXT PRIMARY KEY,charged INTEGER NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY,phase TEXT NOT NULL,kind TEXT NOT NULL,phy_key TEXT NOT NULL,request_sha TEXT NOT NULL,status TEXT NOT NULL,worker TEXT NOT NULL,reserved_at REAL NOT NULL,completed_at REAL,result TEXT,result_sha TEXT,error TEXT)')
            c.execute('CREATE INDEX IF NOT EXISTS events_status ON events(status)')
            wanted = dict(registration_sha256=self.binding, branch=branch, limits=canonical(self.limits),
                          total_cap=str(TOTAL_CAPS[branch]), schema='PACKET_PRECHARGE_V1')
            for key, value in wanted.items():
                previous = c.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
                if previous is not None and previous[0] != value:
                    raise BudgetError('Immutable budget registration changed: ' + key)
                c.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', (key, value))
            for phase in self.limits:
                c.execute('INSERT OR IGNORE INTO counters VALUES (?,0)', (phase,))
            existing = dict(c.execute('SELECT phase,charged FROM counters'))
            if set(existing) != set(self.limits):
                raise BudgetError('Ledger contains unregistered phases')

    @contextlib.contextmanager
    def transaction(self):
        c = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
        try:
            c.execute('PRAGMA busy_timeout=60000')
            c.execute('PRAGMA synchronous=FULL')
            c.execute('BEGIN IMMEDIATE')
            yield c
            c.execute('COMMIT')
        except BaseException:
            if c.in_transaction:
                c.execute('ROLLBACK')
            raise
        finally:
            c.close()

    def register_configuration(self, phy_key, configuration):
        text = canonical(configuration)
        with self.transaction() as c:
            row = c.execute('SELECT definition FROM configurations WHERE phy_key=?', (phy_key,)).fetchone()
            if row is not None and row[0] != text:
                raise BudgetError('Physical configuration changed')
            c.execute('INSERT OR IGNORE INTO configurations VALUES (?,?)', (phy_key, text))

    def _check_inflight(self, c):
        failed = c.execute("SELECT event_id FROM events WHERE status='FAILED' LIMIT 1").fetchone()
        if failed:
            raise BudgetError('Failed charged decode requires registered recovery: ' + failed[0])
        for event_id, encoded in c.execute("SELECT event_id,worker FROM events WHERE status='RESERVED'"):
            owner = json.loads(encoded)
            # Each process runs one sequential decoder callback. A reservation
            # from this same caller on entry is a leftover or reentrant request.
            if same_identity(owner, self.worker):
                raise BudgetError('Current worker has unresolved reservation: ' + event_id)
            try:
                live = self.identity_reader(owner['pid'])
            except (OSError, ValueError, IndexError, KeyError):
                raise BudgetError('Orphaned charged decode requires registered recovery: ' + event_id) from None
            if not same_identity(owner, live):
                raise BudgetError('Reservation owner identity changed: ' + event_id)

    def decode_once(self, phase, event_id, kind, phy_key, request, decode):
        if phase not in self.limits or kind not in ('header', 'body') or not isinstance(event_id, str) or not event_id:
            raise BudgetError('Unknown decode phase/kind/event')
        if phase == 'engineering_reserve':
            raise BudgetError('Engineering reserve requires a separately registered recovery adapter')
        request_sha = digest(request)
        with self.transaction() as c:
            if c.execute('SELECT 1 FROM configurations WHERE phy_key=?', (phy_key,)).fetchone() is None:
                raise BudgetError('Unregistered physical configuration')
            old = c.execute('SELECT phase,kind,phy_key,request_sha,status,result,result_sha FROM events WHERE event_id=?', (event_id,)).fetchone()
            if old is not None:
                if old[:4] != (phase, kind, phy_key, request_sha):
                    raise BudgetError('Decode event identity collision')
                if old[4] != 'COMPLETE':
                    raise BudgetError('Unresolved charged event; no silent retry: ' + event_id)
                result = json.loads(old[5])
                if digest(result) != old[6]:
                    raise BudgetError('Completed actual result was modified')
                # Read-only reuse does not invoke another decoder. It remains
                # available for diagnosis even when a later decode failed.
                return result
            self._check_inflight(c)
            if not same_identity(self.worker, self.identity_reader(self.worker['pid'])):
                raise BudgetError('Calling worker identity changed')
            counts = dict(c.execute('SELECT phase,charged FROM counters'))
            if counts[phase] >= self.limits[phase] or sum(counts.values()) >= TOTAL_CAPS[self.branch]:
                raise BudgetError('Registered phase budget exhausted: ' + phase)
            c.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                      (event_id, phase, kind, phy_key, request_sha, 'RESERVED', canonical(self.worker), time.time(), None, None, None, None))
            c.execute('UPDATE counters SET charged=charged+1 WHERE phase=?', (phase,))
        try:
            result = decode()
            encoded, checksum = canonical(result), digest(result)
        except BaseException as error:
            with self.transaction() as c:
                changed = c.execute("UPDATE events SET status='FAILED',completed_at=?,error=? WHERE event_id=? AND status='RESERVED' AND request_sha=?",
                    (time.time(), type(error).__name__ + ': ' + str(error), event_id, request_sha)).rowcount
                if changed != 1:
                    raise BudgetError('Failure reservation changed') from error
            raise
        with self.transaction() as c:
            changed = c.execute("UPDATE events SET status='COMPLETE',completed_at=?,result=?,result_sha=? WHERE event_id=? AND status='RESERVED' AND request_sha=? AND worker=?",
                (time.time(), encoded, checksum, event_id, request_sha, canonical(self.worker))).rowcount
            if changed != 1:
                raise BudgetError('Decode completion reservation changed')
        return result

    def snapshot(self):
        with self.transaction() as c:
            counts = dict(c.execute('SELECT phase,charged FROM counters'))
            grouped = c.execute('SELECT phase,kind,status,count(*) FROM events GROUP BY phase,kind,status').fetchall()
            observed = {phase:sum(n for p,k,s,n in grouped if p==phase) for phase in self.limits}
            if counts != observed or sum(counts.values()) != sum(row[3] for row in grouped):
                raise BudgetError('Counter/event audit mismatch')
            return dict(schema='PACKET_PRECHARGE_V1', registration_sha256=self.binding, branch=self.branch,
                        total_cap=TOTAL_CAPS[self.branch], limits=self.limits, charged=sum(counts.values()),
                        phase_charged=counts, phase_remaining={p:self.limits[p]-counts[p] for p in counts},
                        development_reserve_transferable=False,
                        counts=[dict(phase=p,kind=k,status=s,count=n) for p,k,s,n in grouped],
                        unresolved=sum(n for p,k,s,n in grouped if s!='COMPLETE'),
                        configurations=c.execute('SELECT count(*) FROM configurations').fetchone()[0])

    def assert_quiescent(self):
        value = self.snapshot()
        if value['unresolved']:
            raise BudgetError('Unresolved/failed charged events remain; preserve and diagnose')
        return value
