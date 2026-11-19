"""CPU differential test of stream construction; conditional FIXTURE probabilities only."""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from profiles import OFF, QUARTER, SIDES, count, public_directory, capacity
from split_prefixes import BASES, FAMILY, SourceReject, SplitPrefixes, select_profile

LEGACY_BLOB = 'c1ac3266d7e08a1d98dba0b57963eabc41072af1'
TOKEN_SHA = '09b9d8104cdb83b945cbb15219b208d706a379f905603eefdc4fa387402f08ec'

def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def reference(path, source_tokens):
    raw = path.read_bytes().replace(b'\r\n', b'\n')
    blob = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    require(blob == LEGACY_BLOB, 'historical split encoder blob differs')
    cls, = [n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name=='SplitSourceCodec']
    fn, = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='encode_all']
    ns = dict(np=np, WHOLE_FAMILY=FAMILY, OFFSETS=OFF, ENDS=QUARTER[:-1], source_tokens=source_tokens)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns['encode_all']

class Provider:
    def __init__(self):
        self.s, self.state, self.ready = 0, 0, False
    def cdf(self, scale):
        require(self.s == scale and not self.ready, 'invalid receiver order')
        self.ready = True
        # Strictly positive conditional CDF, total 2^24. Rotate substantial
        # frequency differences with received coarse tokens; no model invocation.
        f = np.full(4096,4096,np.int64)
        ix = (np.arange(4096)+self.state)%4096
        f[ix[:2048]] += 800; f[ix[2048:]] -= 800
        return np.broadcast_to(np.r_[0,f.cumsum()], (SIDES[scale]**2,4097))
    def advance(self, q):
        require(self.ready and len(q) == SIDES[self.s]**2, 'invalid coarse context')
        self.state = int((self.state*31+np.dot(1+np.arange(len(q)),q)+17)%4096)
        self.s += 1; self.ready = False

class Fixture:
    def __init__(self, enc, dec): self.Encoder, self.Decoder = enc, dec
    def _inference(self, family): return nullcontext()
    def _context(self, family): return nullcontext(Provider())
    def _cdf(self, family, p, s): return p.cdf(s)
    def _table(self, p, s, trace): trace.append(s); return p.cdf(s)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','legacy','tokens','out'): p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args(); require(not a.out.exists(),'use a fresh output file'); start=time.perf_counter(); cpu=time.process_time()
    sys.path.insert(0,str(a.repo/'experiments/wcl-evidence-closure-20261009/scripts'))
    from t1_entropy_core import load_integer_codec,source_tokens
    with np.load(a.tokens,allow_pickle=False) as z: q=source_tokens(z['tokens']).copy()
    h=hashlib.sha256(b'int64:680\0'+q.astype('<i8').tobytes()).hexdigest(); require(h==TOKEN_SHA,'registered source token identity mismatch')
    _,enc,dec=load_integer_codec(a.repo); fixture=Fixture(enc,dec); new=SplitPrefixes(fixture)
    old=reference(a.legacy,source_tokens)(fixture,q); single,base,extra=new.encode(q)
    for tag, streams in (('single',single),('base',base),('enhancement',extra)):
        for key, rec in old[tag].items(): require(np.array_equal(rec['bits'],streams[key]),f'legacy mismatch {tag}:{key}')
    decoded={j:new.decode_base(j,base[j]) for j in BASES}
    for j in BASES: require(np.array_equal(decoded[j],q[:OFF[j]]),'base RX mismatch')
    for (j,m,k),bits in extra.items():
        v=new.decode_enhancement(j,(m,k),decoded[j],bits)
        require(np.array_equal(v,q[OFF[j]:count((m,k))]),'enhancement RX mismatch')
    for bad in (base[5][:-1],np.r_[base[5],np.uint8(0)]):
        try: new.decode_base(5,bad)
        except SourceReject: pass
        else: raise RuntimeError('RX accepted a noncanonical stream')
    class Broken(Fixture):
        def _cdf(self,family,p,s): raise ValueError('deliberate numerical provider failure')
    try: SplitPrefixes(Broken(enc,dec)).decode_base(5,base[5])
    except RuntimeError as err: require(isinstance(err.__cause__,ValueError),'wrong propagated model failure')
    else: raise RuntimeError('model failure became a reception outcome')
    # Independently enumerate selected base, then enhancement, for every declared policy.
    from profiles import BR,EPHY,ALPHA
    d=public_directory(); inv={v:k for k,v in d.items() if k}; checked=0
    for grid in ('QUARTER','WHOLE'):
        ends=[e for e in QUARTER if grid=='QUARTER' or e[1]==0]
        for jcap in (5,6,7):
            for t in (e for e in ends if e[0]>=6 and count(e)>OFF[jcap]):
                for br in BR:
                    for qe,er in EPHY:
                        for ap in ALPHA:
                            obj=select_profile(base,extra,jcap,t,br,qe,er,ap,grid,inv); checked+=1
                            js=[j for j in BASES if j<=jcap and len(base[j])<=capacity(2,br)]
                            if not js: require(obj is None,'missing paid NULL'); continue
                            j=max(js); es=[e for e in ends if OFF[j]<count(e)<=count(t) and len(extra[(j,*e)])<=capacity(qe,er)]
                            best=max(es,key=count) if es else (j,0)
                            require(d[obj[0]]==(j,*best,br,qe,er,ap),'directory/fallback mismatch')
    result=dict(status='PASS_ENGINEERING_CONDITIONAL_CDF_FIXTURE',fixture='scale-dependent integer CDF driven by recovered coarse tokens; NOT the frozen VAR model',registered_source='n01514859/ILSVRC2012_val_00006195_n01514859',token_sha256=h,legacy_encoder_git_blob=LEGACY_BLOB,unchanged_streams={k:len(v) for k,v in old.items() if isinstance(v,dict)},independent_base_decodes=len(decoded),independent_enhancement_decodes=len(extra),new_m10_edges=4,all_target_policy_fallbacks_checked=checked,malformed_stream_rejections=2,model_failure_is_not_gray=True,scientific_lengths_added=0,neural_calls=0,gpu_hours=0,wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu)
    with a.out.open('x',encoding='utf8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
