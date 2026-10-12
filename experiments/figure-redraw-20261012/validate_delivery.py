"""Read-only evidence checks for redraw outputs, without scientific calls."""
from pathlib import Path
import csv, hashlib, json, math, xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/figures/redraw_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    figures=read(OUT/'delivery_index.json');assert all(x['status']=='EXPORTED' for x in figures)
    assert {'M01','M02','M03','M04'}<=set(x['id'] for x in figures)
    numeric=0;vectors=0
    for figure in figures:
        folder=ROOT/figure['path'];provenance=read(folder/'provenance.json')
        assert provenance['bootstrap_calls']==provenance['model_calls']==0
        assert provenance['script_sha256']==sha(ROOT/'scripts/plot_paper_redraw_v1.py')
        for rel,h in provenance['input_bindings'].items():assert sha(ROOT/rel)==h,rel
        rows=list(csv.DictReader((folder/'plot_data.csv').open()))
        for row in rows:
            if not row.get('mean'):continue
            v=float(row['mean']);assert math.isfinite(v)
            if 'ci_low' in row:assert float(row['ci_low'])<=v<=float(row['ci_high'])
            rel=row['input_csv']
            if rel.endswith('.csv'):
                source=list(csv.DictReader((ROOT/rel).open()))
            else:source=read(ROOT/rel)['paired_results']
            compare=[k for k in ['metric','mean','ci_low','ci_high','method','reference','snr_db','source_count','noise_count'] if k in row]
            found=[s for s in source if all(k in s and (float(s[k])==float(row[k]) if k in ['mean','ci_low','ci_high','snr_db','source_count','noise_count'] else str(s[k])==row[k]) for k in compare)]
            assert found,(figure['id'],row)
            numeric+=1
        for name,h in provenance['outputs'].items():
            p=folder/name;assert sha(p)==h,p
            if p.suffix=='.svg':
                tree=ET.parse(p);texts=tree.findall('.//{http://www.w3.org/2000/svg}text');assert texts,p
                if figure['id'] not in ['M04','S05_photos','S16']:
                    assert not tree.findall('.//{http://www.w3.org/2000/svg}image'),p
                vectors+=1
    evidence=ROOT/'results/figure_redraw_20261012/final_examples_v1'
    assert read(evidence/'actual_wait.json')['returncode']==0
    done=read(evidence/'completion.json');receipts=read(evidence/'receipts.json')
    assert done['frames']==8 and sha(evidence/'receipts.json')==done['receipts_sha256']
    assert {(r['method'],r['source_index'],r['snr_db'],r['noise_seed']) for r in receipts}=={
        (m,i,13,2001) for m in ['P100000','Swin200000'] for i in [21,92,95,75]}
    for row in receipts:
        for k in ['png','archive']:assert sha(row[k]['path'])==row[k]['sha256']
    report=dict(status='PASS_READ_ONLY_DELIVERY_VALIDATION',figure_groups=len(figures),
        published_numeric_rows_verified=numeric,editable_svg_files=vectors,
        final_example_actual_exit=0,final_example_frames=8,training_updates=0,bootstrap_calls=0,
        originals_unmodified=True,scientific_calls_in_validator=0,
        visual_inspection='Recorded separately after PNG/PDF inspection')
    target=ROOT/'results/figure_redraw_20261012/delivery_validation.json'
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
