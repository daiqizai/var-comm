"""Apply the preregistered dual-metric extension rule, without reselection."""
from pathlib import Path
import argparse
from uep_common import read,seal,sha,require
from model_validation import rows_csv
from optimize import development_extension_gate

def main():
    p=argparse.ArgumentParser()
    for key in ('policies','scores','validation','output'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();reg=read(a.scores/'registration.json');done=read(a.scores/'completion.json')
    require(done['status']=='UEP_ACTUAL_SCORING_COMPLETE' and done['registration_sha256']==sha(a.scores/'registration.json'),'Actual scoring incomplete')
    table=a.scores/'metrics_per_frame.csv.gz';require(done['outputs'][str(table.resolve())]==sha(table),'Actual metrics changed')
    rows=rows_csv(table);validation=read(a.validation)
    require(validation['input_bindings'][str(a.policies.resolve())]==sha(a.policies),'Model validation uses a different frozen policy')
    require(validation.get('status')=='MODEL_VALIDATION_COMPLETE' and validation.get('synthetic') is False,'Model validation receipt incomplete')
    for path in (table,a.scores/'completion.json',a.scores/'registration.json'):
        require(validation['input_bindings'].get(str(path.resolve()))==sha(path),'Model validation uses different actual scored data')
    source_ids=[v['source_id'] for v in reg['source_identity']]
    result=development_extension_gate(rows,source_ids,read(a.policies),validation['model_validation'])
    result['input_bindings']={str(v.resolve()):sha(v) for v in (a.policies,a.validation,table,a.scores/'completion.json')}
    seal(a.output,result);print(result['status'])

if __name__=='__main__':main()
