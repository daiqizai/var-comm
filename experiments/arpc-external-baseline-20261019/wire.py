"""Standalone ARPC container: paid text, reversible mode/length tags, native bits.

The endpoint (1..7) is supplied by a public paid profile. No image identifiers,
CDFs, hidden Python boundaries, or original tokens are accepted by this API.
"""
import zlib
import numpy as np
from utils.arithmeticcoding import ArithmeticEncoder, ArithmeticDecoder, BitCollector, BitListReader

SIDES = (1,2,4,6,8,12,16)

class SourceParseError(ValueError): pass

def encode_text(text):
    c=zlib.compressobj(level=9,wbits=-15,memLevel=8,strategy=zlib.Z_DEFAULT_STRATEGY)
    compressed=c.compress(text)+c.flush(zlib.Z_FINISH)
    return (1,compressed) if len(compressed)<len(text) else (0,text)

def decode_text(mode,blob):
    if not mode:return blob
    try:
        c=zlib.decompressobj(wbits=-15);text=c.decompress(blob,512)
        if len(text)>511 or not c.eof or c.unused_data or c.unconsumed_tail:
            raise SourceParseError('invalid or oversized caption deflate stream')
        return text
    except zlib.error as e:raise SourceParseError('invalid caption deflate stream') from e

def integer_bits(value,width):
    if type(value) is not int or not 0 <= value < 2**width:raise ValueError((value,width))
    return [(value >> i) & 1 for i in range(width-1,-1,-1)]

def cumulative(counts):
    return np.concatenate((np.zeros(1,dtype=np.uint64),np.cumsum(counts,dtype=np.uint64)))

def encode_tags(tags):
    out=BitCollector();coder=ArithmeticEncoder(32,out);counts=np.ones(16,dtype=np.uint64)
    for tag in tags:
        if not 0<=tag<16:raise ValueError(tag)
        coder.write(cumulative(counts),tag);counts[tag]+=1
    coder.finish()
    return out.bits

def decode_tags(bits,count):
    coder=ArithmeticDecoder(32,BitListReader(bits));counts=np.ones(16,dtype=np.uint64);tags=[]
    for _ in range(count):
        tag=coder.read(cumulative(counts),16);tags.append(tag);counts[tag]+=1
    if encode_tags(tags)!=bits:raise SourceParseError('noncanonical or corrupted tag arithmetic stream')
    return tags

class Reader:
    def __init__(self,data):
        self.bits=[(b>>i)&1 for b in data for i in range(7,-1,-1)];self.pos=0
    def take(self,n):
        if n<0 or self.pos+n>len(self.bits):raise SourceParseError('truncated payload')
        result=self.bits[self.pos:self.pos+n];self.pos+=n;return result
    def uint(self,n):
        result=0
        for b in self.take(n):result=(result<<1)|b
        return result

def pack(caption,trans,help_flags,endpoint):
    if type(endpoint) is not int or not 1<=endpoint<=7:raise ValueError('endpoint')
    text=caption.encode('utf8',errors='strict')
    if not 0<len(text)<=511 or '\x00' in caption:raise ValueError('caption length/content')
    if len(trans)!=endpoint or len(help_flags)!=endpoint:raise ValueError('exact prefix required')
    tags=[];visual=[]
    for side,blocks,modes in zip(SIDES,trans,help_flags):
        if len(blocks)!=side*side or len(modes)!=side*side:raise ValueError('scale shape')
        for block,mode in zip(blocks,modes):
            if mode not in (0,1) or any(b not in (0,1) for b in block):raise ValueError('bits')
            if mode==0:
                if len(block)!=16:raise ValueError('raw block length')
                tag=0
            else:
                if not 1<=len(block)<=15:raise ValueError('native coded block length')
                tag=len(block)
            tags.append(tag);visual.extend(block)
    fixed=[b for tag in tags for b in integer_bits(tag,4)]
    adaptive=encode_tags(tags)
    # Fixed transport selector, determined only by actual bit lengths; no quality feedback.
    mode=int(13+len(adaptive)<len(fixed))
    table=integer_bits(len(adaptive),13)+adaptive if mode else fixed
    text_mode,text_payload=encode_text(text)
    bits=[text_mode]+integer_bits(len(text_payload),9)+[b for byte in text_payload for b in integer_bits(byte,8)]+[mode]+table+visual
    pad=(-len(bits))%8;bits.extend([0]*pad)
    payload=bytes(sum(bits[i+j]<<(7-j) for j in range(8)) for i in range(0,len(bits),8))
    details={'payload_bits':len(bits),'caption_bytes':len(text),'caption_bits':8*len(text_payload),
        'caption_encoding':'deflate_raw_level9' if text_mode else 'utf8_raw','caption_mode_bits':1,
        'caption_wire_bytes':len(text_payload),'caption_uncompressed_bits':8*len(text),
        'caption_length_bits':9,'metadata_mode_bit':1,'mode_and_boundary_bits':len(table),
        'tag_stream_length_bits':13 if mode else 0,'tag_arithmetic_bits_including_finish':len(adaptive) if mode else 0,
        'visual_bits_including_native_finishes':len(visual),'alignment_bits':pad,'tag_encoding':'adaptive16' if mode else 'fixed4',
        'original_visual_plus_help_bits':len(visual)+len(tags),'native_blocks':len(tags),
        'coded_blocks':sum(t!=0 for t in tags),'raw_blocks':sum(t==0 for t in tags)}
    assert len(bits)==details['caption_bits']+1+9+1+len(table)+len(visual)+pad
    return payload,details

def unpack(data,endpoint):
    if type(endpoint) is not int or not 1<=endpoint<=7:raise SourceParseError('unknown endpoint/profile')
    if not isinstance(data,bytes):raise SourceParseError('bytes required')
    r=Reader(data);text_mode=r.uint(1);n=r.uint(9)
    if n==0:raise SourceParseError('empty caption')
    try:caption=decode_text(text_mode,bytes(r.uint(8) for _ in range(n))).decode('utf8',errors='strict')
    except UnicodeError as e:raise SourceParseError('invalid UTF8') from e
    if '\x00' in caption:raise SourceParseError('invalid caption')
    count=sum(s*s for s in SIDES[:endpoint]);mode=r.uint(1)
    if mode:
        length=r.uint(13)
        if length==0 or length>4*count:raise SourceParseError('invalid tag stream length')
        tags=decode_tags(r.take(length),count)
    else:tags=[r.uint(4) for _ in range(count)]
    trans=[];modes=[];j=0
    for side in SIDES[:endpoint]:
        blocks=[];flags=[]
        for _ in range(side*side):
            t=tags[j];j+=1;blocks.append(r.take(t or 16));flags.append(int(t!=0))
        trans.append(blocks);modes.append(flags)
    pad=(-r.pos)%8
    if len(r.bits)-r.pos!=pad or any(r.take(pad)):raise SourceParseError('invalid trailing padding/data')
    # Canonical reserialization is a syntax check on the actual received data.
    # It never consults transmitted truth and accepts altered but well-formed payloads.
    canonical,_=pack(caption,trans,modes,endpoint)
    if canonical!=data:raise SourceParseError('noncanonical metadata format')
    return caption,trans,modes
