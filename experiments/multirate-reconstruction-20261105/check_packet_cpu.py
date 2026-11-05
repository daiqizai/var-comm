"""Wire-format regression checks; no GPU/model inference."""
from runtime import setup
setup()
from codec import pack,unpack
from model import mode_spec
def rejects(w):
    try:unpack(w)
    except (AssertionError,ValueError):return
    raise AssertionError('malformed packet accepted')
def main():
    for mode in range(16):
        w=pack(mode,0);assert len(w)==17 and unpack(w)==(mode,0,b'',b'')
        top=b'abc';escape=b'xyz';assert unpack(pack(mode,1,top,escape))==(mode,1,top,escape)
        bad=bytearray(w);bad[-1]^=1;rejects(bytes(bad));rejects(w+b'x');rejects(w[:-1])
    rejects(pack(0,0,b'x'));rejects(pack(0,61));rejects(pack(0,1,b'x'*100))
    assert mode_spec(0)==(256,512) and mode_spec(8)==(512,2048)
    print('PASS: 16 paid modes, NULL/positive prefix, CRC, exact lengths, capacity and endpoint bounds; zero model calls')
if __name__=='__main__':main()
