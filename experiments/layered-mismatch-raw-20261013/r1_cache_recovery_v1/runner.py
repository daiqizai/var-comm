"""Explicit recovered worker routing, never a global startup hook."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import common as c
import adapter

def run(argv):
    parser=argparse.ArgumentParser();parser.add_argument('--original',required=True)
    args,rest=parser.parse_known_args(argv)
    original=Path(args.original)
    c.require(original in (c.OLD/'receiver_visual.py',c.OLD/'score_selection.py'),'Only missing original worker routes allowed')
    c.require(rest and rest[0] in ('visual','score','select'),'No source/PHY/projection endpoint is exposed')
    role={'visual':'visual','score':'score','select':'selection'}[rest[0]]
    c.require(original==(c.OLD/('receiver_visual.py' if role=='visual' else 'score_selection.py')),'Worker script/action pair differs')
    sha_flag='--expected-registration' if role=='visual' else '--registration-sha256'
    c.require(rest.count('--registration')==rest.count(sha_flag)==1,'Exact explicit registration required')
    registration=dict(path=rest[rest.index('--registration')+1],sha256=rest[rest.index(sha_flag)+1])
    s,receiver,score,_=adapter.install(registration,role)
    sys.argv=[str(original)]+rest
    # Reuse original main() for visual; score has a CLI block, so route the
    # already loaded bound functions directly with identical arguments.
    if role=='visual':return receiver.main()
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--registration',required=True)
    p.add_argument('--registration-sha256',required=True);p.add_argument('--parent',type=int,required=True)
    a=p.parse_args(rest)
    return score.score_worker(registration,a.parent) if role=='score' else score.select(registration)

if __name__=='__main__':
    sys.dont_write_bytecode=True
    print(c.json.dumps(run(sys.argv[1:])),flush=True)
