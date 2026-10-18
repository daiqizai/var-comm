"""Preserve failed pre-registration attempt; one metadata repair, original science."""
import argparse
import json
import os
import pathlib
import signal
import sys
import time
import traceback

E = pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
CODE = E / 'a3_metric_names_v1'
sys.path.insert(0, str(E / 'a3_metrics_v1'))
import control as c
import data
sys.path.insert(0, str(CODE))
from adapter import adapted

def main():
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=['qualify', 'prepare', 'dispatch', '_prepare'])
    p.add_argument('--expected-registration')
    a = p.parse_args()
    oldfail = c.L / 'a3_metrics_prepare_control_v1'
    if a.role == '_prepare':
        q = c.L / 'a3_metric_names_qualify_control_v1'
        c.success(q / 'child.actual_wait.json')
        assert c.gone(json.loads((q / 'supervisor.json').read_text())['process'])
        failed = json.loads((oldfail / 'child.actual_wait.json').read_text())
        assert type(failed['returncode']) is int and failed['returncode'] == 1 and c.gone(failed['process'])
        assert c.gone(json.loads((oldfail / 'supervisor.json').read_text())['process'])
        assert not c.RUN.exists() and not c.OUT.exists()
        data.baseline_rows = adapted((c.CODE / 'data.py').read_bytes())
        original_put = c.put
        def put(path, value):
            if path == c.REG:
                value = dict(value)
                value['source_files'] += [c.desc(f) for f in sorted(CODE.iterdir()) if f.is_file()]
                value['inputs'] += [c.desc(oldfail / n) for n in ['supervisor.json','child.launch.json','child.actual_wait.json','child.stdout','child.stderr','failure.json']]
                value['inputs'] += [c.desc(q / n) for n in ['supervisor.json','child.launch.json','child.actual_wait.json','child.stdout','child.stderr','completion.json']]
                value['metadata_name_revision'] = {'main_table_alias': {'BPG_ADAPTIVE':'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'}, 'original_data':c.desc(c.CODE/'data.py'), 'adapter':c.desc(CODE/'adapter.py'), 'original_failed_preparation_preserved':True, 'scientific_calls_before_revision':0, 'numeric_functions_changed':False}
            original_put(path, value)
        c.put = put
        c.prepare()
        return
    root = c.L / ('a3_metrics_dispatch_control_v1' if a.role == 'dispatch' else 'a3_metric_names_'+a.role+'_control_v1')
    assert not root.exists()
    if a.role == 'qualify':
        argv = [str(c.PY), '-B', str(CODE/'test_alias.py')]
    elif a.role == 'prepare':
        argv = [str(c.PY), '-B', str(CODE/'execute.py'), '_prepare']
    else:
        c.success(c.L/'a3_metric_names_prepare_control_v1/child.actual_wait.json')
        assert c.gone(json.loads((c.L/'a3_metric_names_prepare_control_v1/supervisor.json').read_text())['process'])
        assert c.desc(c.REG)['sha256'] == a.expected_registration
        assert not (c.RUN/'owner.launch.json').exists()
        argv = [str(c.PY), '-B', str(c.CODE/'control.py'), 'run', '--expected-registration', a.expected_registration]
    root.mkdir()
    c.put(root/'supervisor.json', {'process':c.identity(os.getpid()),'parent_process':c.identity(os.getppid()),'argv':sys.argv,'source':c.desc(__file__),'started_unix':time.time()})
    c.old.RUN = root
    os.chdir(CODE)
    try:
        c.old.run_stage('child', argv, dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'), 3000 if a.role=='dispatch' else 600)
        c.put(root/'completion.json', {'status':'PASS','actual_wait':c.desc(root/'child.actual_wait.json')})
    except BaseException:
        if a.role == 'dispatch':
            pending=[]
            for name in ['score','statistics']:
                path=c.RUN/(name+'.launch.json')
                if path.exists():
                    ident=json.loads(path.read_text())['process']
                    if not c.gone(ident):
                        assert os.getpgid(ident['pid'])==ident['pid']
                        os.killpg(ident['pid'],signal.SIGTERM);pending.append(ident)
            if pending:
                time.sleep(5)
                for ident in pending:
                    if not c.gone(ident):os.killpg(ident['pid'],signal.SIGKILL)
        c.put(root/'failure.json', {'error':traceback.format_exc(),'retry_authorized':False})
        raise

if __name__ == '__main__':
    main()
