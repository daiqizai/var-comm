"""Paid predicted-class container; no source/model imports."""
import hashlib
import numpy as np
CLASS_BITS=10
LENGTH_BITS=13
CRC_BITS=16
OVERHEAD_BITS=CLASS_BITS+LENGTH_BITS+CRC_BITS
TAIL=dict(temperature=1.0,top_k=2000,top_p=1.0,cfg_scale=4.0,cfg_interval=-1)
def class_bits(value):
 if type(value)is not int or not 0<=value<1000:raise ValueError('Paid predicted class must be 0..999')
 return np.array([(value>>i)&1 for i in range(9,-1,-1)],np.uint8)
def received_class(bits):
 a=np.asarray(bits)
 if a.dtype!=np.uint8 or a.shape!=(10,)or not np.isin(a,(0,1)).all():raise ValueError('Exactly ten received class bits required')
 value=sum(int(v)<<(9-i)for i,v in enumerate(a))
 if value>=1000:raise ValueError('Reserved received class')
 return value
def generation_seed(bits,prefix,predicted_class):
 b=np.asarray(bits);c=class_bits(predicted_class)
 if type(prefix)is not int or not 1<=prefix<=256 or b.dtype!=np.uint8 or b.ndim!=1 or not np.isin(b,(0,1)).all():raise ValueError('Invalid received generation identity')
 raw=b'ADA_PREDCLASS_FIXEDFRAME_TAIL_V1|'+str(prefix).encode()+b'|'+c.tobytes()+b'|'+b.tobytes()
 return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')%(1<<31)
def source_capacity(k):
 if type(k)is not int or k<=OVERHEAD_BITS:raise ValueError('Invalid information block size')
 return min(k-OVERHEAD_BITS,8191-CLASS_BITS)
def container(predicted_class,arithmetic_bits):
 a=np.asarray(arithmetic_bits)
 if a.dtype!=np.uint8 or a.ndim!=1 or not 2<=len(a)<=8181 or not np.isin(a,(0,1)).all():raise ValueError('Invalid terminated arithmetic vector')
 return np.concatenate((class_bits(predicted_class),a))
def split_container(bits):
 a=np.asarray(bits)
 if a.dtype!=np.uint8 or a.ndim!=1 or not 12<=len(a)<=8191 or not np.isin(a,(0,1)).all():raise ValueError('Invalid received class/arithmetic container')
 return received_class(a[:10]),a[10:].copy()
