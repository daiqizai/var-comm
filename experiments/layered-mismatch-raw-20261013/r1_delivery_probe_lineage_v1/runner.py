"""Explicit CPU entry routing; never a global Python startup hook."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=('dispatcher','delivery','figures','backup','fetch'))
    parser.add_argument('--expected-adapter',required=True)
    parser.add_argument('--expected-runner',required=True)
    parser.add_argument('--expected-registration',required=True)
    parser.add_argument('--authorization-path')
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--authorization-bytes',type=int)
    parser.add_argument('--revision-authorization')
    parser.add_argument('--revision-authorization-sha256')
    parser.add_argument('--revision-authorization-bytes',type=int)
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--sync-light',action='store_true')
    args=parser.parse_args()
    if args.role!='fetch' and (not sys.platform.startswith('linux') or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise RuntimeError('Original Linux CPU interpreter with CUDA hidden required')
    path=Path(__file__).resolve().parent/'adapter.py'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=args.expected_adapter:
        raise RuntimeError('Explicit adapter source SHA differs')
    spec=importlib.util.spec_from_file_location('_r1_cpu_probe_lineage_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    if args.role=='fetch':
        adapter.require(os.name=='nt' and args.expected_registration==adapter.MASTER and args.receipt_sha256,
                        'Windows fetch needs actual server receipt SHA and fixed master')
        fetch=adapter.load_local_fetch(Path(__file__).resolve().parent,args.expected_adapter,args.expected_runner)
        print(json.dumps(fetch.fetch(args.receipt_sha256,args.sync_light)),flush=True)
        return
    adapter.require(all((args.revision_authorization,args.revision_authorization_sha256,args.revision_authorization_bytes)),
                    'Root-pinned metadata-only revision authorization required')
    context=adapter.Context(args.expected_adapter,args.expected_runner,
        dict(path=args.revision_authorization,sha256=args.revision_authorization_sha256,bytes=args.revision_authorization_bytes))
    adapter.require(Path(__file__).resolve()==Path(context.runner['path']),'Runner location differs')
    adapter.require(args.expected_registration==adapter.MASTER,'Frozen recovery master required')
    module=context.load(args.role)
    if args.role=='dispatcher':
        module.run()
    elif args.role in ('delivery','figures'):
        # Only the legacy argument parser receives this view. Actual subprocess
        # argv stays the explicit runner route and is recorded as such.
        sys.argv=[module.__file__,'--expected-registration',adapter.MASTER]
        module.main()
    else:
        adapter.require(all((args.authorization_path,args.authorization_sha256,args.authorization_bytes)),
                        'Actual final closure descriptor required')
        print(json.dumps(module.build(args)),flush=True)


if __name__=='__main__':main()
