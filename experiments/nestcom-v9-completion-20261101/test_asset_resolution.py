"""Meaningful failure-path tests for reader asset relocation. No model work."""
import hashlib,tempfile
from pathlib import Path
from verify_assets import verify
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp);f=p/'tiny.bin';f.write_bytes(b'fixed-reference')
 m=dict(reader_asset_mapping={'tiny.bin':'/unavailable/tiny.bin'},files=[dict(path='/original/tiny.bin',sha256=hashlib.sha256(f.read_bytes()).hexdigest())])
 assert verify(m,root=p)['status']=='PASS'
 assert verify(m,overrides={'tiny.bin':str(f)})['status']=='PASS'
 f.write_bytes(b'changed')
 try:verify(m,root=p)
 except ValueError as e:assert 'mismatch' in str(e)
 else:raise AssertionError('corruption accepted')
 try:verify(m)
 except FileNotFoundError:pass
 else:raise AssertionError('missing asset accepted')
print('PASS: root override, explicit mapping, corrupt rejection, missing rejection')
