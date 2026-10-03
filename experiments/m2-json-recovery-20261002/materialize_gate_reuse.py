"""CPU-only materialization of the original M2 screen-to-gate reuse branch.

No scientific source is edited and no model, tensor, image, or RNG is used.
Every derived row must equal the original frozen calibration AST expression.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

RUN='SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
RECOVERY='M2-JSON-RECOVERY-20261002'
STAGE='full_calibration_gate_reduced_noisy'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    def invalid(value):raise RuntimeError('Nonfinite JSON input: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def native_booleans(value):
    if type(value) is np.bool_:return bool(value)
    if isinstance(value,dict):return {k:native_booleans(v) for k,v in value.items()}
    if isinstance(value,list):return [native_booleans(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def encode(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def verify(bindings):
    if not bindings:raise RuntimeError('Required immutable bindings are empty')
    for path,value in bindings.items():
        if sha(path)!=value:raise RuntimeError('Immutable input changed: '+path)


def atomic_preserve(path,value):
    """Never overwrite a checkpoint, including one created by another process."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if digest(read(path))!=digest(value):raise RuntimeError('Existing checkpoint differs: '+str(path))
        return False
    temporary=path.with_name(path.name+f'.materialize.{os.getpid()}.tmp')
    with temporary.open('xb') as handle:
        handle.write(encode(value));handle.flush();os.fsync(handle.fileno())
    try:
        try:os.link(temporary,path)
        except FileExistsError:
            if digest(read(path))!=digest(value):raise RuntimeError('Concurrent checkpoint differs: '+str(path))
            return False
        return True
    finally:
        temporary.unlink()


def constant(tree,name):
    matches=[n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1
             and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    if len(matches)!=1:raise RuntimeError('Expected one frozen constant: '+name)
    return ast.literal_eval(matches[0])


def function(tree,name,namespace,filename):
    matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(matches)!=1 or matches[0].decorator_list:raise RuntimeError('Unexpected frozen pure function: '+name)
    exec(compile(ast.Module(body=matches,type_ignores=[]),str(filename),'exec'),namespace)
    return namespace[name]


def frozen_functions(runner_path,common_path,receiver_path):
    tree=ast.parse(Path(runner_path).read_text(encoding='utf-8'))
    common=ast.parse(Path(common_path).read_text(encoding='utf-8'))
    receiver=ast.parse(Path(receiver_path).read_text(encoding='utf-8'))
    seeds=constant(common,'CAL_SEEDS');lambdas=constant(tree,'LAMBDAS');count=constant(tree,'CAL_SELECT')
    projections=constant(receiver,'PROJECTIONS')
    if count!=200 or seeds!=[4101,4102,4103]:raise RuntimeError('Original registered calibration scope differs')
    choose=function(tree,'choose_lambda',dict(np=np,defaultdict=defaultdict,METRICS=constant(tree,'METRICS')),runner_path)
    resource=function(receiver,'resource_record',dict(PROJECTIONS=projections),receiver_path)
    calibration=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibration')
    reused=[n for n in ast.walk(calibration) if isinstance(n,ast.Assign) and len(n.targets)==1
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='reused']
    staged=[n.value for n in ast.walk(calibration) if isinstance(n,ast.AugAssign)
            and isinstance(n.target,ast.Name) and n.target.id=='rows' and isinstance(n.value,ast.ListComp)
            and any(isinstance(t,ast.Constant) and t.value==STAGE for t in ast.walk(n.value))]
    if len(reused)!=1 or len(staged)!=1:raise RuntimeError('Original screen-to-gate reuse AST changed')
    expressions=[compile(ast.Expression(value),str(runner_path),'eval') for value in (reused[0].value,staged[0])]
    return dict(choose=choose,resource=resource,seeds=seeds,lambdas=lambdas,count=count,projections=projections,
                reused_expression=expressions[0],stage_expression=expressions[1])


def csv_scalar(value):return '' if value is None else str(value)


def validate_csv(path,expected):
    with Path(path).open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle);actual=list(reader);fields=reader.fieldnames
    if len(actual)!=len(expected) or set(fields or [])!={k for row in expected for k in row}:
        raise RuntimeError('Original CSV row/field coverage differs: '+str(path))
    for before,after in zip(expected,actual):
        if {k:csv_scalar(before.get(k,'')) for k in fields}!=after:
            raise RuntimeError('Original CSV differs from registered source rows: '+str(path))


def derive(screen,functions,names):
    selected=functions['choose'](screen,'VAR_GUIDED')
    static=functions['choose'](screen,'STATIC')
    indexed=defaultdict(list)
    for row in screen:indexed[row['source_index'],row['projection'],row['noise_seed']].append(row)
    outputs={}
    for i in range(functions['count']):
        direct=[];original=[]
        for name in sorted(names):
            lam=selected[name]['selected_lambda']
            for seed in functions['seeds']:
                matches=[r for r in indexed[i,name,seed] if r['control']=='UNGUIDED'
                         or (r['control']=='VAR_GUIDED' and float(r['lambda'])==lam)]
                if len(matches)!=2 or [r['control'] for r in matches]!=['UNGUIDED','VAR_GUIDED']:
                    raise RuntimeError('Screen does not contain the exact ordered original pair')
                direct.extend(dict(row,stage=STAGE) for row in matches)
                namespace=dict(screen=screen,i=i,name=name,seed=seed,lam=lam)
                reused=eval(functions['reused_expression'],namespace)
                if len(reused)!=2:raise RuntimeError('Original calibration would run inference for this pair')
                original.extend(eval(functions['stage_expression'],dict(reused=reused)))
        if digest(direct)!=digest(original):raise RuntimeError('Derived rows differ from original reuse AST')
        outputs[i]=direct
    return outputs,dict(var=native_booleans(selected),static=native_booleans(static))


def prepare(root):
    root=Path(root).resolve();base=root/'experiments/scale-causal-partial-residual-20261002'
    parent=root/'outputs'/RUN;calibration=parent/'m2/calibration'
    registration_path=parent/'m2_calibration_registration.json';registration=read(registration_path)
    registered=registration['source_bindings'];verify(registered)
    runner,common,receiver=[base/name for name in ('m2_runner.py','common.py','residual_receiver.py')]
    for path in (runner,common,receiver):
        if registered.get(str(path))!=sha(path):raise RuntimeError('AST source not in original calibration registration')
    if registration.get('stage')!='m2_calibration' or registration.get('training_updates')!=0:
        raise RuntimeError('Wrong original calibration registration')
    functions=frozen_functions(runner,common,receiver);regsha=sha(registration_path)
    inputs={str(registration_path):regsha,**registered,str(Path(__file__).resolve()):sha(__file__)}
    ledger=root/'results/scale_causal_partial_residual_20261002/m2_resource_ledger.csv'
    ledger_rows=[functions['resource'](N,g,ch,phy) for g,ch in functions['projections']
                 for N in (512,1024) for phy in ('QPSK','16QAM')]
    validate_csv(ledger,ledger_rows);inputs[str(ledger)]=sha(ledger)
    names={f"g{r['g']}_c{r['channels']}" for r in ledger_rows if r['feasible']}
    expected={(f'g{g}_c{ch}',seed,control,lam) for g,ch in functions['projections'] for seed in functions['seeds']
        for control,lam in [('UNGUIDED',0.),('LIKELIHOOD',0.),('DIRECT',0.)]
            +[(control,lam) for control in ('VAR_GUIDED','STATIC') for lam in functions['lambdas']]}
    screen=[]
    for index in range(200):
        path=calibration/'screen'/f'source_{index:04d}.json';body=read(path);inputs[str(path)]=sha(path)
        if set(body)!={'registration_sha256','rows'} or body['registration_sha256']!=regsha:
            raise RuntimeError('Screen source belongs to another calibration')
        rows=body['rows']
        actual={(r['projection'],r['noise_seed'],r['control'],float(r['lambda'])) for r in rows}
        if len(rows)!=len(expected) or actual!=expected:raise RuntimeError('Original complete screen grid required')
        for row in rows:
            if (type(row['source_index']) is not int or row['source_index']!=index
                    or row['source_id']!=registration['source_ids'][index]
                    or row['preprocessing_id']!=registration['preprocessing_ids'][index]
                    or row.get('stage')!='calibration_screen_reduced_noisy' or row.get('snr_db')!=7):
                raise RuntimeError('Screen source/prefix/seed identity differs')
        screen.extend(rows)
    screen_csv=calibration/'screen_per_frame.csv';validate_csv(screen_csv,screen);inputs[str(screen_csv)]=sha(screen_csv)
    outputs,selection=derive(screen,functions,names)
    records={i:dict(registration_sha256=regsha,rows=rows) for i,rows in outputs.items()}
    proof=dict(status='ORIGINAL_GATE_REUSE_QUALIFIED',derived_cache_only=True,fresh_gpu_inference=False,
        images_or_tensors_read=False,torch_imported='torch' in sys.modules,training_updates=0,policy_selection_changed=False,
        source_indices=list(range(200)),rows=sum(len(x) for x in outputs.values()),projection_names=sorted(names),
        calibration_noise_seeds=functions['seeds'],selected_policy=selection,
        comparison='exact canonical JSON equality to original calibration reuse/stage AST for every source',
        registration_sha256=regsha,inputs=inputs,
        expected_cache_payload_sha256={str(calibration/'gate'/f'source_{i:04d}.json'):digest(value) for i,value in records.items()})
    if proof['torch_imported']:raise RuntimeError('Gate cache preparation must remain CPU-only without Torch')
    verify(inputs)
    return calibration,records,proof


def execute(root,mode):
    root=Path(root).resolve();out=root/'outputs'/RECOVERY;out.mkdir(parents=True,exist_ok=True)
    calibration,records,proof=prepare(root)
    qualification=out/'gate_reuse_qualification.json'
    if mode=='qualify':
        atomic_preserve(qualification,proof)
        return proof
    if mode!='apply':raise ValueError('Unknown materialization mode')
    if not qualification.is_file() or digest(read(qualification))!=digest(proof):
        raise RuntimeError('Matching completed dry-run qualification required before apply')
    # Check every existing file before writing any missing source.
    for index,value in records.items():
        path=calibration/'gate'/f'source_{index:04d}.json'
        if path.exists() and digest(read(path))!=digest(value):raise RuntimeError('Existing gate source differs: '+str(path))
    outputs={}
    for index,value in records.items():
        path=calibration/'gate'/f'source_{index:04d}.json';atomic_preserve(path,value)
        outputs[str(path)]=sha(path)
    verify(proof['inputs']);verify(outputs)
    receipt=dict(status='ORIGINAL_GATE_REUSE_MATERIALIZED',derived_cache_only=True,fresh_gpu_inference=False,
        training_updates=0,policy_selection_changed=False,source_indices=list(records),rows=proof['rows'],
        inputs={str(qualification):sha(qualification),**proof['inputs']},outputs=outputs)
    atomic_preserve(out/'materialized_cache.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--mode',choices=('qualify','apply'),required=True)
    args=parser.parse_args();answer=execute(args.root,args.mode)
    print(json.dumps({'status':answer['status'],'sources':len(answer['source_indices']),'rows':answer['rows']}))
