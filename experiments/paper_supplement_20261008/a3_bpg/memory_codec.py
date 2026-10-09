"""ctypes adapter: original BPG algorithm, no file I/O inside encode/decode/fit."""
import ctypes as C
import struct
import zlib
import numpy as np

def png_bytes(rgb):
    a=np.asarray(rgb)
    if a.dtype!=np.uint8 or a.shape!=(256,256,3):raise ValueError('Require HWC uint8 RGB256')
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    raw=b''.join(b'\0'+row.tobytes() for row in a)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',256,256,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b'')

class MemoryBPG:
    def __init__(self,library):
        self.lib=C.CDLL(str(library))
        self.lib.varcomm_bpg_encode_png.argtypes=[C.c_void_p,C.c_size_t,C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
        self.lib.varcomm_bpg_encode_png.restype=C.c_int
        self.lib.varcomm_bpg_decode_rgb.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_void_p),C.POINTER(C.c_int),C.POINTER(C.c_int)]
        self.lib.varcomm_bpg_decode_rgb.restype=C.c_int
        self.lib.varcomm_bpg_free.argtypes=[C.c_void_p];self.lib.varcomm_bpg_free.restype=None

    def encode_png(self,png,qp):
        if type(qp)is not int or not 0<=qp<=51:raise ValueError('QP must be0..51')
        source=C.create_string_buffer(png);out=C.c_void_p();length=C.c_size_t()
        status=self.lib.varcomm_bpg_encode_png(source,len(png),qp,C.byref(out),C.byref(length))
        if status!=0:raise RuntimeError('Actual memory BPG encoder failed: '+str(status))
        try:return C.string_at(out,length.value)
        finally:self.lib.varcomm_bpg_free(out)

    def encode(self,rgb,qp):
        return self.encode_png(png_bytes(rgb),qp)

    def decode(self,data):
        source=C.create_string_buffer(data);out=C.c_void_p();width=C.c_int();height=C.c_int()
        status=self.lib.varcomm_bpg_decode_rgb(source,len(data),C.byref(out),C.byref(width),C.byref(height))
        if status<0:raise RuntimeError('Actual memory BPG decoder execution failed: '+str(status))
        if status>0:return None
        try:return np.frombuffer(C.string_at(out,width.value*height.value*3),dtype=np.uint8).reshape(height.value,width.value,3).copy()
        finally:self.lib.varcomm_bpg_free(out)

    def fit(self,rgb,capacity):
        # This cache is local to one fresh TX call. Every timed call constructs
        # PNG and recomputes its own complete QP search, matching the old codec.
        png=png_bytes(rgb);streams={}
        def measure(qp):
            if qp not in streams:streams[qp]=self.encode_png(png,qp)
            return len(streams[qp])<=capacity
        chosen=None
        if measure(51):
            lo,hi=0,51
            while lo<hi:
                mid=(lo+hi)//2
                if measure(mid):hi=mid
                else:lo=mid+1
            for qp in range(max(0,lo-2),min(51,lo+2)+1):measure(qp)
            chosen=min(q for q,v in streams.items() if len(v)<=capacity)
        return (None if chosen is None else streams[chosen]),dict(
            status='SOURCE_UNFIT' if chosen is None else 'FIT',selected_qp=chosen,capacity_bytes=capacity,
            tested_qp_bytes={str(q):len(v) for q,v in sorted(streams.items())},actual_encode_calls=len(streams),
            search='QP51 feasibility; binary search; boundary +/-2 checks; no cross-call cache')
