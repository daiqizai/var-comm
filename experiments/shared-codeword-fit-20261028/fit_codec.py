"""One new TX traversal; independent RX contexts see only received source bits."""
from fit_common import *
from fit_protocol import endpoints
from same_scale_codec import SameScaleCodec,WHOLE_FAMILY,OFFSETS,InvalidSourceStream,source_tokens

class FitCodec(SameScaleCodec):
    def encode_fit(self,tokens):
        tokens=source_tokens(tokens);whole=self.Encoder();streams={};trace=[];tables={}
        def finish(name,e):streams[name]=np.asarray(e.finish(),np.uint8)
        with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY)as p:
            for s in range(6):
                if s in(4,5):finish(f'm{s}_K0',whole)
                tab=self._table(p,s,trace);tables[f'cdf{s}']=tab.copy();v=tokens[OFFSETS[s]:OFFSETS[s+1]]
                if s==4:
                    for K in range(26):
                        for role,lo,hi in [('A',0,K//2),('B',K//2,K)]:
                            if hi==lo:streams[f'K{K}_{role}']=np.empty(0,np.uint8)
                            else:
                                e=self.Encoder();e.encode(v[lo:hi],tab[lo:hi]);finish(f'K{K}_{role}',e)
                if s>=4:
                    for j in range(len(v)):
                        whole.encode(v[j:j+1],tab[j:j+1])
                        if j+1<len(v):finish(f'm{s}_K{j+1}',whole)
                else:whole.encode(v,tab)
                p.advance(v.copy())
            finish('m6_K0',whole)
        return streams,tables,trace

    def receive_single(self,bits,m,K):
        assert(m,K)in endpoints()
        with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY)as p:
            trace=[];v=self._decode_group(p,np.asarray(bits,np.uint8),0,m,K,trace)
        return dict(C=v[:OFFSETS[m]].tolist(),A=v[OFFSETS[m]:].tolist(),B=[],m=m,K=K,trace=trace)

    def receive_compact(self,K,C,A,B):
        assert 0<=K<=25
        result=dict(m=4,K=K,C=None,A=None,B=None,trace=[],status_C='MISSING',status_A='CONDITION_UNAVAILABLE',status_B='CONDITION_UNAVAILABLE')
        if C is None:return result
        with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY)as p:
            try:result['C']=self._decode_group(p,np.asarray(C,np.uint8),0,4,0,result['trace']).tolist();result['status_C']='ACCEPTED'
            except InvalidSourceStream as e:result['status_C']='AC_REJECT:'+str(e);return result
            if not K:result.update(A=[],B=[],status_A='DISABLED',status_B='DISABLED');return result
            tab=self._table(p,4,result['trace'])
            for role,bits,lo,hi in [('A',A,0,K//2),('B',B,K//2,K)]:
                if lo==hi:result[role]=[];result['status_'+role]='DISABLED';continue
                if bits is None:result['status_'+role]='MISSING';continue
                try:result[role]=self.decode_slice(bits,tab[lo:hi]).tolist();result['status_'+role]='ACCEPTED'
                except InvalidSourceStream as e:result['status_'+role]='AC_REJECT:'+str(e)
        return result
