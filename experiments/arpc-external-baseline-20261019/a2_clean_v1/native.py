"""A2a population-only adapter of the closed, byte-pinned A1 native engine.

No codec/model/number/preprocessing function is replaced. The reversible edits
only enumerate the remaining sources/endpoints and omit duplicate TX renders.
"""
import ast,ctypes,hashlib,json,os,pathlib,sys
ORIGINAL_SHA='29cf445d424c7f26bf8578ded1ba30354c81846f44a18abeb4e3f1dc0074a03c'
EDITS=[
 ("'caption':8,'tokenizer':8,'tx_probability':8,'reference_render':24", "'caption':92,'tokenizer':92,'tx_probability':92,'reference_render':0"),
 ("'rx_prefix':24,'rx_render':24", "'rx_prefix':676,'rx_render':676"),
 ("['a1_first8'];assert len(sources)==8", "['cal100'][8:];assert len(sources)==92"),
 ('if e in (1,4,7):', 'if e in range(1,8):'),
 ("                    result.append({'key':key,'endpoint':e,**render(tokens[:e],e,key,'reference_render')})", '                    # No duplicate native reference image in A2a.'),
 ("put(wires/'jobs.json',jobs)", "put(wires/'new_jobs.json',jobs)"),
 ('assert len(jobs)==24', 'assert len(jobs)==676'),
 ('assert e in (1,4,7)', 'assert e in range(1,8)'),
]
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==ORIGINAL_SHA,'Original native engine changed'
    original=data.decode();text=original
    for old,new in EDITS:
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new,1)
    inverse=text
    for old,new in reversed(EDITS):
        assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==original
    before=ast.parse(original);after=ast.parse(text)
    # Every top-level definition except the population-enumerating main is identical.
    for a,b in zip(before.body,after.body):
        if isinstance(a,ast.FunctionDef) and a.name=='main':continue
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    return text
def main():
    assert sys.argv[1] in ['tx','rx'];i=sys.argv.index('--public')
    cfg=json.loads(pathlib.Path(sys.argv[i+1]).read_text());E=pathlib.Path(cfg['code'])
    sys.path.insert(0,str(E));sys.path.insert(0,str(E/'cuda_recovery_v1'))
    from runtime import check_mapping
    actual=check_mapping(ctypes.CDLL('libcuda.so.1'),cfg['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'))
    with (pathlib.Path(cfg['run'])/(sys.argv[1]+'.cuda_mapping.json')).open('x') as f:
        json.dump({'actual':actual,'expected':cfg['cuda_admission'],'pid':os.getpid()},f,indent=2)
    source=E/'a1_native.py';text=adapt(source.read_bytes())
    exec(compile(text,str(source)+'::a2_population_v1','exec'),{'__name__':'__main__','__file__':str(source)})
if __name__=='__main__':main()
