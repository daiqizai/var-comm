"""Authorized normal push; GCM credential stays in process memory only."""
from pathlib import Path
import argparse
import base64
import json
import os
import subprocess
import time
from invoke import args_for,read_pin

GIT=r'C:\Users\11946\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
HOST='DevMachineModel-H100-4C-P1660-01'
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--descriptor-pin',required=True);a=p.parse_args()
    d=read_pin(a.descriptor_pin)
    if(HERE/'local_push_attempt.json').exists():raise RuntimeError('Local push already attempted; inspect remote state')
    clean={k:v for k,v in os.environ.items()if not k.upper().startswith(('GIT_TRACE','GCM_TRACE'))
        and k.upper()not in('GIT_CURL_VERBOSE','GCM_DEBUG')}
    env=dict(clean,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    result=subprocess.run([GIT,'credential','fill'],input='protocol=https\nhost=github.com\n\n',
        env=env,capture_output=True,text=True,check=True)
    fields=dict(line.split('=',1)for line in result.stdout.splitlines()if'='in line)
    auth=base64.b64encode((fields['username']+':'+fields['password']).encode()).decode()
    forward=subprocess.Popen(['ssh','-N','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30',
        '-R','127.0.0.1:23098:127.0.0.1:7897',HOST],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        time.sleep(2)
        if forward.poll()is not None:raise RuntimeError('Own reverse proxy unavailable; no push attempted')
        with(HERE/'local_push_attempt.json').open('x',encoding='utf8')as f:
            json.dump(dict(descriptor=d,started_unix=time.time(),normal_push=True),f,indent=2)
        result=subprocess.run(args_for('push',d),input=json.dumps({'auth':auth}),text=True,
            encoding='utf8',capture_output=True,timeout=600,env=env)
        if result.returncode:
            raise RuntimeError('Remote push failed or uncertain; inspect fixed remote receipts; no automatic retry')
        receipt=json.loads(result.stdout)
        if receipt.get('normal_push')is not True or receipt['head']!=receipt['verified_remote']or receipt['descriptor']!=d:
            raise RuntimeError('Push response is not an exact remote-SHA confirmation')
        with(HERE/'local_push_verified.json').open('x',encoding='utf8',newline='\n')as f:
            json.dump(receipt,f,indent=2);f.write('\n')
        print(json.dumps(receipt))
    finally:
        if forward.poll()is None:forward.terminate()
        forward.wait(timeout=15)
        # No credential-bearing command, input, environment or stderr is echoed.


if __name__=='__main__':main()
