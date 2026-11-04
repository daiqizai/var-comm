"""CPU protocol checks; never reported as neural codec qualification."""
import json
from packet import SourcePacket,choose_longest_feasible

def main():
    packets=[SourcePacket(0,b'',b''),SourcePacket(1,b'1234',b''),SourcePacket(2,b'12345678',b'1234'),SourcePacket(3,b'1234',b'')]
    for p in packets:
        wire=p.to_bytes();assert SourcePacket.from_bytes(wire)==p
        for i in range(len(wire)):
            corrupt=bytearray(wire);corrupt[i]^=1
            try:SourcePacket.from_bytes(corrupt)
            except ValueError:pass
            else:raise AssertionError('corrupt packet accepted')
        for bad in [wire[:-1],wire+b'\x00']:
            try:SourcePacket.from_bytes(bad)
            except ValueError:pass
            else:raise AssertionError('noncanonical length accepted')
    assert choose_longest_feasible(packets,104).slices==3 # deliberate nonmonotone lengths
    assert choose_longest_feasible(packets,72).slices==0
    print(json.dumps(dict(cpu_packet_checks=True,monotonicity_assumption=False,neural_codec_qualified=False)))

if __name__=='__main__':main()
