"""Read-only constructor audit of the exact selected original tables, CPU only."""
import argparse
import importlib
from pathlib import Path
import traceback
from history_common import write, source_bindings, normalize_metadata, identity

ROSTER = [
    ('historical_selected_budget','FINAL_P2048_P3060',3000),
    ('historical_selected_budget','FINAL_P4084_SELECTED_SEEDS',3000),
    ('historical_selected_budget','FINAL_DIGITAL_QPSK',9000),
    ('historical_selected_budget','FINAL_DIGITAL_16QAM',9000),
    ('historical_selected_budget','FINAL_PHASE2_N4084_DIGITAL',3000),
    ('historical_cached','SELECTED_EXTERNAL_AUTHORS',13500),
    ('historical_cached','LEGACY_N3060_FINAL',4500),
    ('historical_selected_references','SELECTED_NOISELESS_REFERENCES',1800),
    ('historical_selected_optional','OPTIONAL_RX_STEP2_A',2400),
    ('historical_selected_optional','OPTIONAL_RX_STEP2_B',2400),
    ('historical_selected_optional','OPTIONAL_H6_13DB',600),
    ('historical_selected_optional','OPTIONAL_PHASE2_MAIN',4500),
]


def main(root, output):
    root=Path(root).resolve(); results=[]; proofs={}
    for module, study, count in ROSTER:
        try:
            a=importlib.import_module(module).create_adapter(root,study)
            rows=[r for i in range(100) for r in a.expected_rows(i)]
            sources={r.get('image_id',r.get('source_id')) for r in rows}
            if len(sources)!=100 or None in sources or len(rows)!=count:
                raise RuntimeError('Selected count differs: '+str((len(sources),len(rows),count)))
            methods=set()
            for row in rows:
                methods.add(normalize_metadata(row,a.metadata(row))['method_id'])
            result=dict(study=study,adapter=module,status='PASS',sources=100,frames=count,
                methods=sorted(methods),selected_rows_sha256=identity(rows),description=a.describe())
            proofs.update(a.bindings)
        except Exception as error:
            result=dict(study=study,adapter=module,status='FAIL',error=str(error),traceback=traceback.format_exc())
        results.append(result)
        print(study,result['status'],result.get('error',''),flush=True)
        write(output,dict(status='RUNNING',scientific_result=False,GPU=False,studies=results))
    write(output,dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',
        scientific_result=False,GPU=False,studies=results,input_proof_bindings=proofs,
        source_bindings=source_bindings(Path(__file__).parent)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
