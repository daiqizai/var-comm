"""Actual received streams only; preserve the qualified native mathematical engine."""
import ast,ctypes,hashlib,json,os,pathlib,sys
ORIGINAL_SHA='29cf445d424c7f26bf8578ded1ba30354c81846f44a18abeb4e3f1dc0074a03c'
EDITS=[
 ("choices=['tx','rx']", "choices=['rx']"),
 ("'rx_prefix':24,'rx_render':24", "'rx_prefix':cfg['new_job_count'],'rx_render':cfg['new_job_count']"),
 ('from wire import pack,unpack,SIDES', 'from wire import pack,unpack,SIDES,SourceParseError'),
 ('assert len(jobs)==24', "assert len(jobs)==cfg['new_job_count'] and len(jobs)<=3600"),
 ('assert e in (1,4,7)', 'assert e in range(1,8)'),
 ("raise RuntimeError('Native source block canonical recoding failed')", "raise SourceParseError('Native source block canonical recoding failed')"),
 ("            def receive():\n                with torch.inference_mode():return decode(args,prior,vae,schedule,caption,text_tokenizer,text_encoder,None,trans,modes)",
  "            source_error=[]\n            def receive():\n                try:\n                    if len(text_tokenizer(caption,truncation=False).input_ids)>512:raise SourceParseError('Received caption exceeds frozen text input limit')\n                    with torch.inference_mode():return decode(args,prior,vae,schedule,caption,text_tokenizer,text_encoder,None,trans,modes)\n                except SourceParseError as error:\n                    source_error.append(str(error));return None"),
 ("            assert len(tokens)==e and len(cdfs)==e",
  "            if tokens is None:\n                assert len(source_error)==1\n                result.append({'key':key,'endpoint':e,'status':'SOURCE_PARSE_FAIL','source_error':source_error[0]})\n                continue\n            assert len(tokens)==e and len(cdfs)==e"),
 ("            caption,trans,modes=unpack(pathlib.Path(job['path']).read_bytes(),e)",
  "            pin(job['input'])\n            caption,trans,modes=unpack(pathlib.Path(job['path']).read_bytes(),e)"),
]
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==ORIGINAL_SHA
    original=data.decode();text=original
    for old,new in EDITS:
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new,1)
    inverse=text
    for old,new in reversed(EDITS):
        assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==original
    before=ast.parse(original);after=ast.parse(text)
    for a,b in zip(before.body,after.body):
        if isinstance(a,ast.FunctionDef) and a.name=='main':continue
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    return text
def main():
    assert sys.argv[1]=='rx';i=sys.argv.index('--public');cfg=json.loads(pathlib.Path(sys.argv[i+1]).read_text())
    E=pathlib.Path(cfg['code']);sys.path.insert(0,str(E));sys.path.insert(0,str(E/'cuda_recovery_v1'))
    from a1_owner import put
    from runtime import check_mapping
    actual=check_mapping(ctypes.CDLL('libcuda.so.1'),cfg['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'))
    put(pathlib.Path(cfg['run'])/'rx.cuda_mapping.json',{'actual':actual,'expected':cfg['cuda_admission'],'pid':os.getpid()})
    source=E/'a1_native.py';text=adapt(source.read_bytes())
    if cfg['new_job_count']==0:
        out=pathlib.Path(cfg['run'])/'rx';out.mkdir()
        put(out/'completion.json',{'status':'COMPLETE','counts':{},'outputs':[],'blocked_reads':[],'model_calls':0})
        return
    exec(compile(text,str(source)+'::actual_received_source_v1','exec'),{'__name__':'__main__','__file__':str(source)})
if __name__=='__main__':main()
