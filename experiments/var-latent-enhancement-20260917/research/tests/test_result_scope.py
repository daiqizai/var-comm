import csv,importlib.util,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[4]
def load_tool(name,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def rows():
    return [dict(method=m,source_index=i,image_id=f'image{i}',snr_db=s,seed=n,N=4084,E=8168,decoder_sha256='dc',
                 protocol_id=m,checkpoint_sha256=m,model_context_sha256=m,psnr_db=20,lpips_alex=.1,dino_cosine=.8)
            for m in ('a','b') for i in range(2) for s in (1,13) for n in (2001,2002,2003)]
@pytest.mark.parametrize('damage',['duplicate','missing','image','decoder','budget','model_context','nan'])
def test_summarizer_rejects_invalid_pairing(tmp_path,monkeypatch,damage):
    tool=load_tool('summarize_review_runs',monkeypatch);r=rows()
    if damage=='duplicate':r.append(dict(r[0]))
    elif damage=='missing':r.pop()
    elif damage=='image':r[-1]['image_id']='another-image'
    elif damage=='decoder':r[-1]['decoder_sha256']='different'
    elif damage=='budget':r[-1]['N']=4096
    elif damage=='model_context':r[-1]['model_context_sha256']='changed-policy'
    elif damage=='nan':r[-1]['lpips_alex']=float('nan')
    path=tmp_path/'rows.csv';tool.write_csv(path,r)
    with pytest.raises(RuntimeError):tool.summarize(path,tmp_path/'analysis')
def test_collector_rejects_common_missing_cells(monkeypatch):
    tool=load_tool('collect_research_comparison',monkeypatch)
    # Equal left/right subsets are not the complete registered development grid.
    with pytest.raises(RuntimeError,match='incomplete'):
        tool.require_grid(rows(),2)
