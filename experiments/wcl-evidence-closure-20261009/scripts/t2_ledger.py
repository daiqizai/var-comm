"""Durable bounded packet callbacks for new WCL runs; never opens old ledgers.

Interface: Ledger(new_path, request_hash, cap).call(event, request, callback).
An exact COMPLETE event returns its saved result without running the callback.
An unresolved reservation is preserved and blocks automatic repeat.
"""
import json
from pathlib import Path
import sqlite3

def require(ok,message):
    if not ok:raise RuntimeError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

class Ledger:
    def __init__(self,path,request_hash,cap):
        require(type(cap)is int and cap>0,'Positive independent cap required')
        require(len(request_hash)==64,'Explicit request SHA required')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.cap=cap;self.path=path
        self.db=sqlite3.connect(path,timeout=90,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, hash TEXT NOT NULL, cap INTEGER NOT NULL)')
        self.db.execute('INSERT OR IGNORE INTO config VALUES (1,?,?)',(request_hash,cap))
        if self.db.execute('SELECT hash,cap FROM config WHERE id=1').fetchone()!=(request_hash,cap):
            self.db.close();raise RuntimeError('Ledger request/cap changed')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, event TEXT NOT NULL, request TEXT NOT NULL, result TEXT, status TEXT NOT NULL)')
    def call(self,event,request,fn):
        eid=event['event_id'];ev=canonical(event);req=canonical(request)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT event,request,result,status FROM events WHERE event_id=?',(eid,)).fetchone()
            if row:
                require(row[0]==ev and row[1]==req and row[3]=='COMPLETE','Unresolved/changed packet; no automatic retry: '+eid)
                self.db.execute('COMMIT');return json.loads(row[2])
            require(self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]<self.cap,'Independent packet cap exhausted')
            self.db.execute('INSERT INTO events VALUES (?,?,?,NULL,?)',(eid,ev,req,'RESERVED'));self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
        # Reservation is durable before the actual decoder is invoked.
        value=fn();self.db.execute('UPDATE events SET result=?,status=? WHERE event_id=?',(canonical(value),'COMPLETE',eid));return value
    def snapshot(self):
        return dict(total=self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0],
            unresolved=self.db.execute("SELECT COUNT(*) FROM events WHERE status!='COMPLETE'").fetchone()[0],cap=self.cap)
    def close(self):self.db.close()
