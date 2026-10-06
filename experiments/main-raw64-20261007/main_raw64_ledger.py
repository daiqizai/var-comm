"""Independent MAIN raw64 packet ledger. All caps come from the sealed plan."""
import hashlib
import json
from contextlib import contextmanager, closing
from pathlib import Path
import sqlite3
import time

SCHEMA = 'MAIN_RAW64_LEDGER_V1'
PHASES = ('qualification', 'proxy')
RESERVED_PHASES = ('actual_calibration', 'development', 'raw_holdout', 'online_PHY', 'optional_H_holdout')


def valid_caps(caps):
    require(set(caps) in (set(PHASES), set(PHASES + RESERVED_PHASES)) and all(type(v) is int and v >= 0 for v in caps.values()), 'Prospective exact CPU/all-study phase caps required')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Ledger:
    def __init__(self, path, registration_sha256, caps):
        self.path = str(path)
        self.registration = registration_sha256
        self.caps = dict(caps)
        valid_caps(self.caps)
        require(Path(path).is_file(), 'Ledger must be explicitly created by registered owner')
        with self.connect() as db:
            try:
                meta = dict(db.execute('SELECT key,value FROM metadata'))
            except sqlite3.DatabaseError as error:
                raise ValueError('Not a new MAIN raw64 ledger') from error
            require(meta == {'schema': SCHEMA, 'registration': registration_sha256, 'caps': canonical(self.caps)}, 'Legacy or differently registered ledger forbidden')

    @classmethod
    def create(cls, path, registration_sha256, caps):
        valid_caps(caps)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb'):
            pass
        with closing(sqlite3.connect(str(p))) as db, db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.executemany('INSERT INTO metadata VALUES (?,?)', [('schema', SCHEMA), ('registration', registration_sha256), ('caps', canonical(caps))])
            db.execute('CREATE TABLE phase_counts (phase TEXT PRIMARY KEY,charged INTEGER NOT NULL CHECK(charged>=0))')
            db.executemany('INSERT INTO phase_counts VALUES (?,0)', [(p,) for p in caps])
            db.execute('''CREATE TABLE events (
                event_id TEXT PRIMARY KEY, phase TEXT NOT NULL,kind TEXT NOT NULL,phy_key TEXT NOT NULL,
                request TEXT NOT NULL,request_sha TEXT NOT NULL,worker TEXT NOT NULL,
                status TEXT NOT NULL,result TEXT,result_sha TEXT,error TEXT,charged_at REAL NOT NULL)''')
            db.execute('CREATE INDEX event_status ON events(status)')
            db.execute('CREATE INDEX event_phase ON events(phase)')
        return cls(path, registration_sha256, caps)

    @contextmanager
    def connect(self):
        db = sqlite3.connect('file:' + Path(self.path).absolute().as_posix() + '?mode=rw', uri=True, timeout=60)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def decode_once(self, event, request, decode, worker):
        phase = event['phase']
        require(phase in PHASES and event['kind'] in ('header', 'body'), 'Unimplemented or unregistered phase/kind')
        body = canonical(request)
        request_sha = digest(request)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT * FROM events WHERE event_id=?', (event['event_id'],)).fetchone()
            if prior is not None:
                require(prior['phase'] == phase and prior['kind'] == event['kind'] and prior['phy_key'] == event['phy_key'] and prior['request'] == body and prior['request_sha'] == request_sha, 'Changed event cannot reuse a charge')
                require(prior['status'] == 'COMPLETE', 'Failed or unresolved event cannot retry')
                value = json.loads(prior['result'])
                require(digest(value) == prior['result_sha'], 'Stored result hash changed')
                return value
            require(not db.execute("SELECT 1 FROM events WHERE status='FAILED' LIMIT 1").fetchone(), 'Failed ledger blocks further decoding')
            require(db.execute('SELECT charged FROM phase_counts WHERE phase=?', (phase,)).fetchone()[0] < self.caps[phase], 'Phase budget exhausted; no borrowing')
            require(db.execute("SELECT count(*) FROM events WHERE status='RESERVED'").fetchone()[0] < 2, 'At most two active packet callbacks')
            db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (event['event_id'], phase, event['kind'], event['phy_key'], body, request_sha, canonical(worker), 'RESERVED', None, None, None, time.time()))
            db.execute('UPDATE phase_counts SET charged=charged+1 WHERE phase=?', (phase,))
        try:
            value = decode()
            encoded = canonical(value)
            result_sha = digest(value)
        except BaseException as error:
            with self.connect() as db:
                db.execute("UPDATE events SET status='FAILED',error=? WHERE event_id=? AND status='RESERVED'", (type(error).__name__ + ': ' + str(error), event['event_id']))
            raise
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute("UPDATE events SET status='COMPLETE',result=?,result_sha=? WHERE event_id=? AND status='RESERVED'", (encoded, result_sha, event['event_id'])).rowcount
            require(changed == 1, 'Prepaid event state changed')
        return value

    def events(self, phase=None):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM events' + (' WHERE phase=?' if phase else '') + ' ORDER BY event_id', (phase,) if phase else ()).fetchall()
        return {row['event_id']: dict(row) for row in rows}

    def snapshot(self):
        with self.connect() as db:
            db.execute('BEGIN')
            rows = db.execute('SELECT phase,status,count(*) FROM events GROUP BY phase,status').fetchall()
            counters = dict(db.execute('SELECT phase,charged FROM phase_counts'))
        charged = {p: 0 for p in self.caps}
        counts = []
        for phase, status, n in rows:
            require(phase in charged and status in ('RESERVED', 'FAILED', 'COMPLETE'), 'Unexpected ledger event')
            charged[phase] += n
            counts.append({'phase': phase, 'status': status, 'count': n})
        require(charged == counters and all(charged[p] <= self.caps[p] for p in charged), 'Transactional counters disagree with paid events')
        return {'registration_sha256': self.registration, 'caps': self.caps, 'phase_charged': charged, 'charged': sum(charged.values()), 'remaining': {p: self.caps[p] - charged[p] for p in PHASES}, 'unresolved': sum(x['count'] for x in counts if x['status'] != 'COMPLETE'), 'counts': sorted(counts, key=lambda x: (x['phase'], x['status']))}

    def quiescent(self):
        value = self.snapshot()
        require(value['unresolved'] == 0, 'All charged events must normally complete')
        return value
