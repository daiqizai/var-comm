"""Existing bounded process owner, one exact role and immutable launch request."""
import argparse,json,sys
from pathlib import Path
from runtime import setup,sha
setup()
from owned_process import run_role

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();r=json.loads(Path(a.request).read_text())
    bindings=dict(r['bindings']);bindings[str(Path(a.request).resolve())]=sha(a.request)
    run_role(r['root'],r['role'],r['command'],r['env'],r['cpus'],r['gpus'],r['max_seconds'],bindings)

if __name__=='__main__':main()
