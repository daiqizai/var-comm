"""Versioned source packet; all serialized bytes count toward source-bit budget.

The RDVQ top and escape streams include their own rANS termination states.
This wrapper adds endpoint, boundaries and CRC, not another entropy state.
Resolution, model, CDF settings and completion are shared protocol context.
"""
from dataclasses import dataclass
import struct
import zlib

HEADER = struct.Struct('>BHH')
CRC = struct.Struct('>I')
VERSION = 1
PACKET_OVERHEAD_BITS = 72

@dataclass(frozen=True)
class SourcePacket:
    slices: int
    top: bytes
    escape: bytes

    def to_bytes(self):
        if not 0 <= self.slices <= 28:
            raise ValueError('slice endpoint outside the frozen 0..28 catalogue')
        if self.slices == 0 and (self.top or self.escape):
            raise ValueError('NULL must have no entropy payload')
        if self.slices > 0 and len(self.top) < 4:
            raise ValueError('nonempty prefix requires a terminated top stream')
        if self.escape and len(self.escape) < 4:
            raise ValueError('escape stream is missing its termination state')
        body = HEADER.pack((VERSION << 5) | self.slices, len(self.top), len(self.escape)) + self.top + self.escape
        return body + CRC.pack(zlib.crc32(body))

    @classmethod
    def from_bytes(cls, wire):
        if len(wire) < HEADER.size + CRC.size:
            raise ValueError('truncated source packet')
        tag, nt, nr = HEADER.unpack_from(wire)
        if tag >> 5 != VERSION or (tag & 31) > 28:
            raise ValueError('unknown packet version or endpoint')
        if len(wire) != HEADER.size + nt + nr + CRC.size:
            raise ValueError('noncanonical packet length')
        if zlib.crc32(wire[:-4]) != CRC.unpack(wire[-4:])[0]:
            raise ValueError('source CRC mismatch')
        p = cls(tag & 31, wire[5:5+nt], wire[5+nt:-4])
        if p.to_bytes() != wire:
            raise ValueError('noncanonical source packet')
        return p

    def accounting(self):
        top_state = 32 if self.top else 0
        escape_state = 32 if self.escape else 0
        return dict(packet_bits=8*len(self.to_bytes()), entropy_stream_bits=8*(len(self.top)+len(self.escape)),
                    version_endpoint_bits=8, stream_boundary_bits=32, source_crc_bits=32,
                    rans_state_bits=top_state+escape_state,
                    entropy_renormalization_bits=8*(len(self.top)+len(self.escape))-top_state-escape_state,
                    top_stream_bits=8*len(self.top), escape_stream_bits=8*len(self.escape),
                    per_slice_header_bits=0, phy_bits=0)

def choose_longest_feasible(packets, budget_bits):
    """Enumerate actual lengths; do not assume rANS lengths are monotone."""
    feasible = [p for p in packets if len(p.to_bytes())*8 <= budget_bits]
    if not feasible:
        if budget_bits < PACKET_OVERHEAD_BITS:
            raise ValueError('budget cannot carry the registered NULL packet')
        return SourcePacket(0,b'',b'')
    return max(feasible,key=lambda p:p.slices)

