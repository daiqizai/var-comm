"""Independent CPU fixture qualification of the exact development source code.

This does not claim a real-model qualification. Actual model identity, complete
source closure, flags, cached token equivalence and canonical decoding remain
mandatory checks in the separately registered source execution.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback
import development_source_common as c


def validate_request(cfg):
    c.require(cfg['schema']=='H_DEVELOPMENT_SOURCE_CPU_QUALIFICATION_REQUEST_V1' and sys.platform.startswith('linux'),
        'Bound Linux CPU fixture qualification required')
    c.verify(cfg['source_bindings']);c.verify(cfg['input_bindings'])
    directory=Path(cfg['code_dir']);required=('development_source_common.py','development100_assets.py',
        'development100_source_driver.py','test_development_source.py','qualify_development_source.py','h_development_driver.py',
        'register_development_source.py','test_register_development_source.py')
    for name in required:
        p=str(directory/name);c.require(cfg['source_bindings'].get(p)==c.sha(p),'Unbound fixture/actual source entry: '+name)
    for name in ('entropy.py','whole_entropy.py'):
        p=str(Path(cfg['reference_var_comm'])/name)
        c.require(c.merge(cfg['source_bindings'],cfg['input_bindings']).get(p)==c.sha(p),'Original arithmetic primitives unbound')
    c.require(Path(cfg['python']).is_absolute() and Path(cfg['python']).is_file() and len(set(cfg['cpu_affinity']))==2
        and set(cfg['cpu_affinity'])<=os.sched_getaffinity(0) and 0<cfg['max_seconds']<=1800,
        'Qualification CPU scope differs')
    return directory


def run(path):
    cfg=c.read(path);directory=validate_request(cfg);out=Path(cfg['out'])
    c.claim(out,c.sha(path),'synthetic_cpu_qualification')
    try:
        os.sched_setaffinity(0,set(cfg['cpu_affinity']));os.setpriority(os.PRIO_PROCESS,0,15)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',H64_REFERENCE_VAR_COMM=cfg['reference_var_comm'],
            **{k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        log=out/'tests.log';argv=[cfg['python'],'-B','-m','unittest','-v','test_development_source','test_register_development_source']
        with log.open('xb') as f:
            result=subprocess.run(argv,cwd=directory,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=cfg['max_seconds'],check=False)
        c.require(result.returncode==0,'Development source CPU qualification failed; evidence preserved')
        c.verify(cfg['source_bindings']);c.verify(cfg['input_bindings'])
        done=dict(status='H_DEVELOPMENT100_SOURCE_QUALIFICATION_PASS',source_bindings=cfg['source_bindings'],
            input_bindings=c.merge(cfg['input_bindings'],{str(Path(path).resolve()):c.sha(path)}),outputs={str(log):c.sha(log)},
            results=[dict(argv=argv,exit_code=0,closed_log_sha256=c.sha(log))],synthetic=True,
            real_model_qualification=False,real_model_execution_checks_required=True,new_packet_decodes=0,
            GPU_used=False,development_read=False,holdout_used=False)
        c.save(out/'completion.json',done);return done
    except BaseException:
        c.save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),new_packet_decodes=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args();print(run(a.config)['status'])
