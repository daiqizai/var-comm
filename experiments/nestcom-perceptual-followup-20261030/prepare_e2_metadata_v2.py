"""Independent metadata-only correction after zero-model planning failure.

Keep the original failed source, output and wait receipt unchanged. The sole
correction is the canonical zero-padded key used by the frozen Ada manifest.
"""
from followup_common import *
p=E/'prepare_e2.py'
assert sha(p)=='5df5dd410d350a3a7cb34a98cd8f71d3a974921e122f9a377c6e852d6f61bfd8'
old="s['ada_payload_hashes'][f'p{i}']";new="s['ada_payload_hashes'][f'p{i:03d}']"
text=p.read_text();assert text.count(old)==1
exec(compile(text.replace(old,new),str(p)+'::CANONICAL_METADATA_KEY_V2','exec'),globals())
