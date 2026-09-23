from types import SimpleNamespace
import numpy as np,pytest,torch
from token_efficiency.digital_grid import candidate_cells,population
from token_efficiency.evaluation_io import load_cell,save_cell
from token_efficiency.execution import prepare_digital,transmit_prepared,transmit,Cell
from token_efficiency.codec import SIZES

def test_candidate_capacity_is_paid_before_channel():
    for mcs in ('QPSK','16QAM'):
        cells,rows=candidate_cells(mcs)
        assert len(rows)==30 and len({r['method'] for r in rows})==30
        assert {c.name for c in cells}=={r['method'] for r in rows if r['status']=='ELIGIBLE'}
        for r in rows:
            assert r['N_header']+r['N_data']==r['N']
            if r['family']=='raw':assert (r['status']=='ELIGIBLE')==(r['raw_source_bits']+22<=r['coded_slots'])
    assert not any(c.family=='raw' and c.m==10 for c in candidate_cells('QPSK')[0])
    assert any(c.family=='raw' and c.m==10 for c in candidate_cells('16QAM')[0])

def test_holdout_is_not_a_population_option():
    with pytest.raises(ValueError,match='no holdout'):population('holdout')

def test_cell_resume_rejects_tamper_scope_and_preserves_unsealed_partial(tmp_path):
    p=tmp_path/'cell.json';r={'registration_sha256':'reg','source_id':'a','method':'m','rows':[]}
    save_cell(p,r);assert load_cell(p,'reg','a','m')==r
    with pytest.raises(RuntimeError,match='scope'):load_cell(p,'wrong','a','m')
    with pytest.raises(RuntimeError,match='overwritten'):save_cell(p,r)
    p.write_text('{}')
    with pytest.raises(RuntimeError,match='hash'):load_cell(p,'reg','a','m')
    other=tmp_path/'partial.json';other.write_text('interrupted')
    assert load_cell(other,'reg','a','m') is None
    assert not other.exists() and next((tmp_path/'uncommitted').glob('*')).read_text()=='interrupted'

def test_offline_preparation_and_complete_online_tx_use_identical_phy():
    quant=SimpleNamespace(f_to_idxBl_or_fhat=lambda F,to_fhat:[torch.zeros((1,s*s),dtype=torch.long) for s in SIZES])
    vae=SimpleNamespace(encoder=lambda x:x,quant_conv=lambda x:x,quantize=quant)
    pixels=np.zeros((3,256,256),np.uint8)
    encoded=prepare_digital(pixels,17,'raw',vae,None,'cpu')
    for mcs in ('QPSK','16QAM'):
        c=Cell('raw',2048,6,mcs)
        a,al=transmit(pixels,17,c,vae,None,'cpu');b,bl=transmit_prepared(encoded,17,c)
        np.testing.assert_array_equal(a,b);assert al==bl
