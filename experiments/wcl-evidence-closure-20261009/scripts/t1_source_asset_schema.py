"""Read/write schema for verified entropy source assets and exact RX cache hits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,require,sha
from t1_codec_runtime import MODEL_STATES

def payload_sha(bits):
    a=np.asarray(bits,dtype=np.uint8)
    return hashlib.sha256(b'uint8_bits\0'+a.tobytes()).hexdigest()

def token_sha(tokens):
    a=np.asarray(tokens,dtype='<i8')
    return hashlib.sha256(f'int64:{len(a)}\0'.encode()+a.tobytes()).hexdigest()

def model_id(family,static_completion_sha):
    if family=='EC_STATIC_WHOLE':return static_completion_sha
    require(family=='EC_VAR_WHOLE','Unknown source family')
    return hashlib.sha256(json.dumps(dict(models=MODEL_STATES,null_class=1000,cdf_total=1<<24,
        entropy_precision=32,ordering='raster',cfg=False),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def build_record(*,source_index,source_id,tokens,preprocessing_id,source_assets_checkpoint,archive,
                 arrays,static_completion_sha,origins,upstream_evidence):
    streams={}
    for family,prefix in [('EC_STATIC_WHOLE','static'),('EC_VAR_WHOLE','var')]:
        per={}
        for m in range(4,10):
            key=f'{prefix}_m{m}_bits'
            if key not in arrays:continue
            bits=np.asarray(arrays[key]);rx_key=f'{prefix}_m{m}_received_tokens'
            require(bits.ndim==1 and bits.dtype==np.uint8 and len(bits)>=2 and np.isin(bits,(0,1)).all(),
                    'Actual bit stream required')
            require(rx_key in arrays and np.array_equal(arrays[rx_key],tokens[:OFFSETS[m]]),
                    'Independent-roundtrip cached output absent/different')
            per[str(m)]=dict(bits_key=key,payload_bits=len(bits),payload_sha256=payload_sha(bits),
                received_tokens_key=rx_key,received_tokens_sha256=token_sha(arrays[rx_key]),
                receiver_cache_proof=dict(independent_roundtrip=True,canonical=True,zero_extension_reads=30,
                    origin=origins[family],source_model_id=model_id(family,static_completion_sha),
                    reuse_requires_exact_received_bits=True,upstream=upstream_evidence[family]))
        streams[family]=per
    return dict(schema='T1_SOURCE_ASSET_V1',source_index=source_index,source_id=source_id,
        source_tokens_sha256=token_sha(tokens),preprocessing_id=preprocessing_id,
        source_assets_checkpoint=source_assets_checkpoint,archive=dict(path=str(Path(archive).resolve()),sha256=sha(archive)),
        tokens_key='tokens',streams=streams,static_model_completion_sha256=static_completion_sha,
        source_role='calibration',online_timing_cache_use_forbidden=True)

def exact_received_cache(record,arrays,*,family,m,received_bits,expected_source_model_id):
    """A miss returns None; never looks at original source tokens or source ID."""
    require(record['schema']=='T1_SOURCE_ASSET_V1','Unexpected source asset schema')
    entry=record['streams'].get(family,{}).get(str(m))
    if entry is None:return None
    proof=entry['receiver_cache_proof']
    require(proof['independent_roundtrip'] is True and proof['canonical'] is True
            and proof['zero_extension_reads']==30,'Unqualified source decode cache')
    if proof['source_model_id']!=expected_source_model_id:return None
    bits=np.asarray(received_bits)
    require(bits.ndim==1 and np.isin(bits,(0,1)).all(),'Invalid received bit vector')
    if len(bits)!=entry['payload_bits'] or payload_sha(bits)!=entry['payload_sha256']:return None
    if not np.array_equal(bits,arrays[entry['bits_key']]):return None
    recovered=np.asarray(arrays[entry['received_tokens_key']],dtype=np.int64)
    require(recovered.shape==(OFFSETS[m],) and token_sha(recovered)==entry['received_tokens_sha256'],
            'Cached independent RX tokens changed')
    return dict(received_tokens=recovered.copy(),source_status='ARITHMETIC_SOURCE_DECODED',
        canonical=True,zero_extension_reads=30,new_probability_calls=0,source_decode_cache_hit=True,
        cache_match='family_m_source_model_and_exact_actual_received_bits',receiver_truth_used=False)
