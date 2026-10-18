import random,unittest
from wire import pack,unpack,SIDES,SourceParseError,encode_tags,decode_tags,decode_text

class WireTests(unittest.TestCase):
    def example(self,e,kind='mixed'):
        rng=random.Random(260101+e);t=[];h=[]
        for side in SIDES[:e]:
            modes=[0 if kind=='raw' else 1 if kind=='coded' else rng.randrange(2) for _ in range(side*side)]
            h.append(modes);t.append([[rng.randrange(2) for _ in range(rng.randrange(1,16) if m else 16)] for m in modes])
        return t,h
    def test_all_endpoints_and_modes(self):
        for e in range(1,8):
            for kind in ['mixed','raw','coded']:
                with self.subTest(endpoint=e,kind=kind):
                    t,h=self.example(e,kind);b,d=pack('a small red bird',t,h,e)
                    self.assertEqual(unpack(b,e),('a small red bird',t,h))
                    self.assertEqual(d['payload_bits'],len(b)*8)
                    self.assertLessEqual(d['mode_and_boundary_bits'],4*d['native_blocks'])
    def test_unicode(self):
        t,h=self.example(4);b,_=pack('一只鸟, café',t,h,4);self.assertEqual(unpack(b,4)[0],'一只鸟, café')
    def test_compressed_text_and_limit(self):
        import zlib
        t,h=self.example(1);b,d=pack('x'*511,t,h,1)
        self.assertEqual(d['caption_encoding'],'deflate_raw_level9')
        self.assertEqual(unpack(b,1)[0],'x'*511)
        c=zlib.compressobj(wbits=-15);blob=c.compress(b'x'*512)+c.flush()
        with self.assertRaises(SourceParseError):decode_text(1,blob)
        with self.assertRaises(SourceParseError):decode_text(1,blob[:-1])
    def test_every_byte_truncation_rejected(self):
        t,h=self.example(4);b,_=pack('an image',t,h,4)
        for n in range(len(b)):
            with self.assertRaises(SourceParseError):unpack(b[:n],4)
    def test_extra_bytes_rejected(self):
        t,h=self.example(1);b,_=pack('a dog',t,h,1)
        with self.assertRaises(SourceParseError):unpack(b+b'\x00',1)
    def test_invalid_caption_length(self):
        t,h=self.example(1)
        for s in ['', 'x'*512,'a\x00b']:
            with self.assertRaises(ValueError):pack(s,t,h,1)
    def test_invalid_endpoint(self):
        for e in [0,8,-1,True]:
            with self.assertRaises(SourceParseError):unpack(b'abc',e)
    def test_native_raw_size_enforced(self):
        with self.assertRaises(ValueError):pack('x',[[[1]*15]],[[0]],1)
    def test_native_compressed_size_enforced(self):
        for n in [0,16,17]:
            with self.assertRaises(ValueError):pack('x',[[[1]*n]],[[1]],1)
    def test_no_hidden_scales(self):
        t,h=self.example(4)
        with self.assertRaises(ValueError):pack('x',t,h,1)
    def test_adaptive_tags(self):
        for tags in [[0]*521,[1]*521,list(range(16))*32,[i%3 for i in range(521)]]:
            self.assertEqual(decode_tags(encode_tags(tags),len(tags)),tags)
    def test_actual_changed_valid_data_is_accepted(self):
        t,h=self.example(1,'raw');b,_=pack('a cat',t,h,1)
        t[0][0][0]^=1;c,_=pack('a cat',t,h,1)
        self.assertNotEqual(b,c);self.assertEqual(unpack(c,1)[1],t)
    def test_invalid_utf8_rejected(self):
        t,h=self.example(1);b,_=pack('x',t,h,1);x=bytearray(b)
        # First raw byte of caption spans bits 10..17.
        bits=[(v>>k)&1 for v in x for k in range(7,-1,-1)];bits[10:18]=[1]*8
        bad=bytes(sum(bits[i+j]<<(7-j) for j in range(8)) for i in range(0,len(bits),8))
        with self.assertRaises(SourceParseError):unpack(bad,1)

if __name__=='__main__':unittest.main()
