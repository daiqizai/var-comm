import ast,hashlib,importlib.util,json,pathlib,sys
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
OLD=R/'experiments/arpc-common500-extension-20261019'
PE=R/'experiments/arpc-external-baseline-20261019'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def adapt_source(raw,adapter):
    original=adapter.adapt(raw)
    text=original.replace("'caption':400,'tokenizer':400,'tx_probability':400", "'caption':24,'tokenizer':24,'tx_probability':24").replace("['new_sources'];assert len(sources)==400", "['new_sources'];assert len(sources)==24")
    assert text!=original and "len(sources)==24" in text
    return text

def adapt_rx(raw,adapter):
    original=adapter.adapt(raw);assert original.count('len(jobs)<=3600')==1
    return original.replace('len(jobs)<=3600','len(jobs)<=216')

def main():
    role=sys.argv[1];assert role in ['tx','rx']
    cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--public')+1]).read_bytes())
    sys.path.insert(0,str(PE));sys.path.insert(0,str(PE/'cuda_recovery_v1'))
    import ctypes,os
    from runtime import check_mapping
    from a1_owner import put
    actual=check_mapping(ctypes.CDLL('libcuda.so.1'),cfg['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'))
    put(pathlib.Path(cfg['run'])/(role+'.cuda_mapping.json'),{'actual':actual,'pid':os.getpid()})
    if role=='rx' and cfg['new_job_count']==0:
        out=pathlib.Path(cfg['run'])/'rx';out.mkdir();put(out/'completion.json',{'status':'COMPLETE','counts':{},'outputs':[],'blocked_reads':[],'model_calls':0});return
    path=PE/'a1_native.py';raw=path.read_bytes()
    adapter=module(OLD/('source_native.py' if role=='tx' else 'rx_native.py'),'frozen_arpc_adapter')
    code=adapt_source(raw,adapter) if role=='tx' else adapt_rx(raw,adapter)
    exec(compile(code,str(path)+'::Kodak24_population_v1','exec'),{'__name__':'__main__','__file__':str(path)})
if __name__=='__main__':main()
