"""Exact reversible I/O adaptation of the completed cal100 DINO worker."""
import ast,hashlib,sys
from pathlib import Path

ORIGINAL_SHA='dd90bd108e3c36c7aff78eb4219a76a616e06c9e7e5c68ec67d85bee54e2beb0'
EDITS=(
 ("regpath=O/'cal100_score_registration.json'", "regpath=O/'b2b3_analysis_v1/B2/score_registration.json'"),
 ("root=L/'cal100_scores_v1'", "root=L/'b2b3_analysis_v1/B2/dino'"),
 ("reference_features='Exact old cal100 DINO features, source archive SHA checked'", "reference_features='Exact old cal1000 DINO features; 100+900 existing, source archive SHA checked'"),
 ("    with np.load(verify(row['image']),allow_pickle=False)as z:a=z[row['array_key']].copy()", "    if row['image'] is None:a=np.full((3,256,256),.5,np.float32)\n    else:\n     with np.load(verify(row['image']),allow_pickle=False)as z:a=z[row['array_key']].copy()"),
)

def adapt(data):
    assert hashlib.sha256(data).hexdigest()==ORIGINAL_SHA
    original=data.decode();text=original
    for old,new in EDITS:
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new,1)
    inverse=text
    for old,new in reversed(EDITS):
        assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==original
    # All model calls, requested metric, numeric flags and reference preparation
    # remain identical. Only path/population text and the established gray image
    # loader differ; no score arithmetic is inserted.
    a,b=ast.parse(original),ast.parse(text)
    calls=lambda tree:[ast.dump(n,include_attributes=False) for n in ast.walk(tree)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('score','MetricEvaluator','as_tensor','set_num_threads','set_num_interop_threads')]
    assert calls(a)==calls(b)
    return text

def main():
    from analysis_common import old,O,L,sha,read,verify,put,pin,dino_identity
    assert len(sys.argv)==2
    p=O/'B2/score_registration.json';assert sha(p)==sys.argv[1]
    r=read(p)
    for d in r['code']:verify(d)
    if r['new_DINO_pairs']==0:
        assert read(verify(r['plan']))==[]
        out=L/'B2/dino/worker';out.mkdir()
        metadata=read(verify(r['prior_metadata']))['DINO_L_metadata']
        put(out/'metric_identity.json',dict(original_DINO_metadata=dino_identity(metadata),current_DINO_metadata=dino_identity(metadata),zero_missing_pairs=True,new_model_loads=0))
        put(out/'scores.json',[])
        put(out/'completion.json',dict(status='NEW_DINO_ONLY_COMPLETE',scores=pin(out/'scores.json'),ledger=dict(caps=dict(model_load=0,reconstruction_dino=0),counts=[],unresolved=0),metric_identity=pin(out/'metric_identity.json'),new_source_features=0,new_VAR=0,new_channel=0,new_bootstrap=0))
        return
    source=old.E/'cal100_dino.py'
    exec(compile(adapt(source.read_bytes()),str(source)+'::B2_actual_pixels_v1','exec'),{'__name__':'__main__','__file__':str(source)})

if __name__=='__main__':main()
