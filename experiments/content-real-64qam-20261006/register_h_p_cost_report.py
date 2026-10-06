"""Seal one CPU report-export stage after original H/P and timing normal exits.

Does not launch any process or read quality/cost tables. Qualification and launch
are explicit external operations; the worker uses the separately frozen driver.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
import h_p_cost_report_driver as d

REGISTERED='H_P_COST_REPORT_REGISTERED_NOT_LAUNCHED'
FILES=('h_p_cost_report.py','test_h_p_cost_report.py','h_p_cost_report_driver.py',
       'test_h_p_cost_report_driver.py','register_h_p_cost_report.py','test_register_h_p_cost_report.py')


def qualify_receipt(request_path,request,x):
    qpath=request['qualification'];q=d.read(qpath)
    for k in ('input_bindings','source_bindings','outputs'):d.verify(q[k])
    argv=[x['python'],'-B','-m','unittest','-v',*d.QUALIFICATION_MODULES]
    d.require(q['status']==d.QUALIFIED and q['python']==x['python'] and q['GPU_used'] is False
        and q['new_packet_decodes']==0 and q['actual_quality_rows_read'] is False
        and q['test_count']==d.QUALIFICATION_COUNT and
        q['results']==[dict(argv=argv,exit_code=0,test_count=d.QUALIFICATION_COUNT)],'Exact CPU interface qualification required')
    d.require(q['input_bindings'].get(str(request_path))==d.sha(request_path),'Qualification must bind this exact registration request')
    for p,s in x['sources'].items():d.require(q['source_bindings'].get(p)==s,'Qualification source differs')
    for p,s in x['bindings'].items():d.require(q['input_bindings'].get(p)==s,'Qualification predecessor differs')
    code=Path(__file__).parent
    for name in FILES:
        p=str(code/name);d.require(x['sources'].get(p)==q['source_bindings'].get(p)==d.sha(p),'All six report sources/tests must be qualified')
    d.require(q['budget_before']==q['budget_after']==x['before'],'Qualification budget differs')
    log=Path(qpath).parent/'tests.log';text=log.read_text(encoding='utf-8',errors='replace')
    d.require(q['outputs'].get(str(log))==d.sha(log) and f'Ran {d.QUALIFICATION_COUNT} tests' in text
        and text.rstrip().endswith('OK'),'Actual successful qualification log required')
    d.require(not (Path(qpath).parent/'failure.json').exists(),'Qualification failure preserved; no implicit retry')
    return q


def make_configs(request,x):
    e=Path(request['execution_dir']);out=Path(request['out']);own=copy.deepcopy(x['template_owner'])
    maximum=request.get('max_seconds',3600)
    d.require(type(maximum)is int and 0<maximum<=7200,'Bounded report stage required')
    cfg=dict(schema='H_P_COST_REPORT_CONFIG_V1',out=str(out),request=str(e/'report_request.json'),
        registration=str(e/'execution_registration.json'),owner_config=str(e/'owner_config.json'),
        qualification=request['qualification'],qualification_test_count=d.QUALIFICATION_COUNT,max_seconds=maximum)
    own.update(owner_out=str(e),registration=cfg['registration'])
    code=str(Path(__file__).parent)
    env=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=code)
    own['stages']=[dict(id='report',resource='cpu',requires=[],max_seconds=maximum,jobs=[dict(id=d.JOB,
        argv=[x['python'],'-B',str(Path(d.__file__).absolute()),'--config',str(e/'report_config.json')],
        cwd=own['root'],out=str(out),completion=str(out/'completion.json'),accepted_statuses=[d.DONE],
        receipt_expect=copy.deepcopy(d.EXPECTED),environment=env)])]
    return cfg,own


def fresh_outputs(request,x):
    e,out=Path(request['execution_dir']),Path(request['out']);base=Path(x['template_owner']['out']).resolve()
    for p in (e,out):
        d.require(p.is_absolute() and base in p.resolve().parents and not p.exists(),'Fresh independent output below original H root required')
    d.require(e.resolve()!=out.resolve() and e.resolve() not in out.resolve().parents and out.resolve() not in e.resolve().parents,
              'Execution and report outputs must be separate siblings or disjoint descendants')
    for b in x['batches'].values():
        for old in (Path(b['data_dir']).resolve(),Path(b['closed']['owner']['owner_out']).resolve()):
            d.require(old not in e.resolve().parents and old not in out.resolve().parents,'Preserve original output directories')


@d.cached_operation
def register(request_path):
    d.require(sys.platform.startswith('linux'),'Linux original-process closure required for actual registration')
    request_path=Path(request_path).absolute();request=d.read(request_path)
    d.require(request['schema']=='H_P_COST_REPORT_REGISTRATION_REQUEST_V1','Explicit report registration request required')
    x=d.preflight(request['report_request']);q=qualify_receipt(request_path,request,x);fresh_outputs(request,x)
    d.unchanged(x);e=Path(request['execution_dir']);e.mkdir(parents=True)
    try:
        cfg,own=make_configs(request,x);cp=e/'report_config.json';op=e/'owner_config.json';rp=e/'execution_registration.json'
        d.save(cfg['request'],request['report_request']);d.save(cp,cfg);d.save(op,own)
        inputs=d.merge(x['bindings'],q['input_bindings'],q['outputs'],d.bind((request_path,request['qualification'],cfg['request'],cp,op)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=e.name,source_stage_scope=d.SCOPE,
            allowed_stage_ids=['report'],owner_config_sha256=d.sha(op),phase_limits=own['phase_limits'],
            budget_registration_sha256=own['budget_registration_sha256'],budget_before=x['before'],
            source_bindings=x['sources'],input_bindings=inputs,created_unix=time.time(),
            CPU_threads=2,GPU_jobs=0,new_packet_decodes=0,new_bootstrap_replicates=0,new_comparisons=0,
            policy_selection=False,MAIN_complete=False,H_success_evaluated=False,future_stage_automatic=False)
        x['owner'].validate_config(own,reg,d.sha(op));d.verify(inputs);d.unchanged(x);d.save(rp,reg)
        d.registered_context(cp,admitted=x)
        result=dict(status=REGISTERED,registration_sha256=d.sha(rp),owner_config_sha256=d.sha(op),
            source_bindings=x['sources'],input_bindings=inputs,outputs=d.bind((cfg['request'],cp,op,rp)),
            budget=x['before'],workers_started=False,GPU_used=False,new_packet_decodes=0,
            actual_quality_rows_read=False,future_stage_automatic=False)
        d.save(e/'registration_completion.json',result);return result
    except BaseException:
        d.save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
    done=register(a.request);print(json.dumps(dict(status=done['status'],workers_started=False)))
