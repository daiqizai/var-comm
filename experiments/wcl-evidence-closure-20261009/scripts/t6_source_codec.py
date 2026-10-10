"""Independent ten-scale source extension for N2048; T1 files stay frozen.

Uses unchanged probability/provider/integer primitives. The paid 13-bit source
length field still caps a transmitted stream at8191 bits independently of k.
"""
import numpy as np
from t1_codec_runtime import SourceCodec,InvalidSourceStream
from t1_entropy_core import OFFSETS,source_tokens,require
from t1_source_asset_schema import build_record as base_record,payload_sha,token_sha,model_id

MAX_SOURCE_LENGTH=8191

class SourceCodec10(SourceCodec):
    def encode(self,family,tokens,modes=(10,)):
        a=source_tokens(tokens);modes=tuple(modes)
        require(modes and tuple(sorted(set(modes)))==modes and all(type(m)is int and 4<=m<=10 for m in modes),
            'Ordered T6 complete prefix modes m4..10 required')
        if max(modes)<10:return super().encode(family,a,modes)
        encoder=self.Encoder();result={}
        with self._inference(family),self._context(family) as provider:
            for scale in range(10):
                values=a[OFFSETS[scale]:OFFSETS[scale+1]];cdf=self._cdf(family,provider,scale)
                encoder.encode(values,cdf)
                if provider is not None:provider.advance(values.copy())
                if scale+1 in modes:
                    bits=np.asarray(encoder.finish(),dtype=np.uint8)
                    result[scale+1]=dict(bits=bits,raw_bits=int(12*OFFSETS[scale+1]),arithmetic_bits=len(bits),flush_bits=len(bits)-len(encoder.bits))
        return result

    def decode(self,family,payload_bits,m):
        require(type(m)is int and 4<=m<=10,'T6 complete prefixes m4..10 required')
        if m<10:return super().decode(family,payload_bits,m)
        a=np.asarray(payload_bits)
        if a.ndim!=1 or len(a)<2 or not np.isin(a,(0,1)).all():raise InvalidSourceStream('Invalid arithmetic bit vector')
        a=a.astype(np.uint8,copy=True)
        class Bounded(self.Decoder):
            def read_bit(inner):
                if inner.position>=len(inner.bits)+30:raise InvalidSourceStream('Arithmetic terminal lookahead exceeded')
                return super().read_bit()
        decoder=Bounded(a);canonical=self.Encoder();recovered=[]
        with self._inference(family),self._context(family) as provider:
            for scale in range(10):
                cdf=self._cdf(family,provider,scale)
                try:values=decoder.decode(cdf)
                except InvalidSourceStream:raise
                except ValueError as error:raise InvalidSourceStream(str(error)) from error
                canonical.encode(values,cdf);recovered.append(values.copy())
                if provider is not None:provider.advance(values.copy())
        if not np.array_equal(canonical.finish(),a):raise InvalidSourceStream('Noncanonical arithmetic termination or length')
        if decoder.position-len(a)!=30:raise InvalidSourceStream('Unexpected terminal lookahead')
        return dict(received_tokens=np.concatenate(recovered).astype(np.int64,copy=False),source_status='ARITHMETIC_SOURCE_DECODED',
            canonical=True,zero_extension_reads=30,transmitted_source_bits=len(a),source_family=family,m=10,K=0,
            receiver_truth_used=False,transmitter_probability_table_used=False)

def build_record(**kwargs):
    """Same exact-RX cache schema with explicitly qualified m10 entries added."""
    record=base_record(**kwargs);arrays=kwargs['arrays'];tokens=kwargs['tokens']
    for family,prefix in [('EC_STATIC_WHOLE','static'),('EC_VAR_WHOLE','var')]:
        key=prefix+'_m10_bits';rx_key=prefix+'_m10_received_tokens'
        if key not in arrays:continue
        bits=np.asarray(arrays[key]);require(bits.ndim==1 and bits.dtype==np.uint8 and len(bits)>=2 and np.isin(bits,(0,1)).all(),'Actual m10 binary stream required')
        require(rx_key in arrays and np.array_equal(arrays[rx_key],tokens),'Actual independent ten-scale roundtrip required')
        record['streams'][family]['10']=dict(bits_key=key,payload_bits=len(bits),payload_sha256=payload_sha(bits),
            received_tokens_key=rx_key,received_tokens_sha256=token_sha(arrays[rx_key]),
            receiver_cache_proof=dict(independent_roundtrip=True,canonical=True,zero_extension_reads=30,
                origin=kwargs['origins'][family],source_model_id=model_id(family,kwargs['static_completion_sha']),
                reuse_requires_exact_received_bits=True,upstream=kwargs['upstream_evidence'][family]))
    record.update(registered_prefixes=list(range(4,11)),source_extension='T6_TEN_SCALE_SAME_SOURCE_MODEL',
        transmitted_length_field_bits=13,max_transmitted_arithmetic_bits=8191)
    return record

def qualify_m10(codec,family,tokens,old_streams=None):
    """Explicit actual TX plus independent RX; caller owns admission/accounting."""
    old_streams={} if old_streams is None else old_streams
    modes=tuple(sorted(set(old_streams)|{10}));encoded=codec.encode(family,tokens,modes)
    for m,bits in old_streams.items():require(np.array_equal(encoded[m]['bits'],bits),'Extended TX changed an existing m4..9 prefix stream')
    decoded=codec.decode(family,encoded[10]['bits'],10)
    require(np.array_equal(decoded['received_tokens'],source_tokens(tokens)),'Independent m10 roundtrip differs')
    return encoded[10],decoded,dict(status='T6_M10_EXACT_SOURCE_QUALIFICATION_PASS',prefix_streams_byte_exact=sorted(old_streams),
        actual_m10_tokens=680,actual_TX_traversals=1,actual_independent_RX_calls=1,null_class=1000,cdf_total=1<<24,
        arithmetic_state_bits=32,canonical=decoded['canonical'],zero_extension_reads=decoded['zero_extension_reads'],
        arithmetic_bits=encoded[10]['arithmetic_bits'],fits_uint13_length=encoded[10]['arithmetic_bits']<=8191,
        raw_substitution=False,source_truth_used_by_receiver=False)
