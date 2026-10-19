"""400 missing-source TX passes only; byte-pinned author mathematics unchanged."""
import ast,ctypes,hashlib,json,os,pathlib,sys
ORIGINAL_SHA='29cf445d424c7f26bf8578ded1ba30354c81846f44a18abeb4e3f1dc0074a03c'
EDITS=[
 ("choices=['tx','rx']", "choices=['tx']"),
 ("'caption':8,'tokenizer':8,'tx_probability':8,'reference_render':24", "'caption':400,'tokenizer':400,'tx_probability':400,'reference_render':0"),
 ("['a1_first8'];assert len(sources)==8", "['new_sources'];assert len(sources)==400"),
 ('if e in (1,4,7):', 'if False:  # No reference generation or duplicate clean RX jobs in A3 TX.'),
]
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==ORIGINAL_SHA
    text=original=data.decode()
    for a,b in EDITS:assert text.count(a)==1;text=text.replace(a,b,1)
    back=text
    for a,b in reversed(EDITS):assert back.count(b)==1;back=back.replace(b,a,1)
    assert back==original
    before=ast.parse(original);after=ast.parse(text)
    for a,b in zip(before.body,after.body):
        if isinstance(a,ast.FunctionDef) and a.name=='main':continue
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    return text
def main():
    assert sys.argv[1]=='tx';cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--public')+1]).read_text());E=pathlib.Path(cfg['code'])
    sys.path.insert(0,str(E));sys.path.insert(0,str(E/'cuda_recovery_v1'));from runtime import check_mapping
    mapping=check_mapping(ctypes.CDLL('libcuda.so.1'),cfg['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'))
    with (pathlib.Path(cfg['run'])/'tx.cuda_mapping.json').open('x') as f:json.dump({'actual':mapping,'expected':cfg['cuda_admission'],'pid':os.getpid()},f,indent=2)
    source=E/'a1_native.py';exec(compile(adapt(source.read_bytes()),str(source)+'::COMMON500_TX400_population_v1','exec'),{'__name__':'__main__','__file__':str(source)})
if __name__=='__main__':main()
