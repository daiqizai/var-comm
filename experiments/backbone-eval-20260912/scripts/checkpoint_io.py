"""Restricted loading of the exact public XQ training checkpoint files.

Torch 2.11 weights-only cannot APPENDS a ruamel CommentedSeq even when its
class is allowlisted. We therefore use a fixed-global-whitelist Unpickler for
these two SHA-verified public artifacts. This is NOT unrestricted pickle:
find_class never dynamically imports or resolves an unlisted name. Only
PyTorch's tensor-safe globals plus seven reviewed config-data classes exist.
The weights_only=False flag is needed to supply this custom pickle_module;
it does not select Python's default/unrestricted Unpickler.
"""
import argparse
import hashlib
import io
from pathlib import Path
import pickle
import pickletools
import types
import zipfile
import torch

EXPECTED = {
    3534394346: 'fa349cec5f953e41a30d1877344fba7e485d5d059fb24b0de83d547aa0c70e53',
    5537116322: '37db545025e1421d91f46e28733d37ef81b0178e2ce229ecbd7c3afcfe4c7c31',
}

def _globals():
    if torch.__version__ != '2.11.0+cu128':
        raise RuntimeError('Restricted-loader tensor-global mapping requires audited torch2.11.0+cu128')
    from ruamel.yaml.anchor import Anchor
    from ruamel.yaml.comments import CommentedSeq, Format, LineCol
    from ruamel.yaml.scalarfloat import ScalarFloat
    from ruamel.yaml.tag import Tag
    from torch._weights_only_unpickler import _get_allowed_globals
    defaults = dict(_get_allowed_globals())
    reviewed = {'argparse.Namespace': argparse.Namespace,
        'ruamel.yaml.anchor.Anchor': Anchor, 'ruamel.yaml.comments.CommentedSeq': CommentedSeq,
        'ruamel.yaml.comments.Format': Format, 'ruamel.yaml.comments.LineCol': LineCol,
        'ruamel.yaml.scalarfloat.ScalarFloat': ScalarFloat, 'ruamel.yaml.tag.Tag': Tag}
    return defaults, reviewed

def _module():
    defaults, reviewed = _globals()
    allowed = {**defaults, **reviewed}
    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            key = module + '.' + name
            if key not in allowed:
                raise pickle.UnpicklingError('Rejected unreviewed global: ' + key)
            return allowed[key]
    return types.SimpleNamespace(__name__='xq_reviewed_tensor_metadata_pickle',
                                 Unpickler=RestrictedUnpickler, load=None)

def load_xq_checkpoint(path):
    path = Path(path)
    want = EXPECTED.get(path.stat().st_size)
    if want is None: raise RuntimeError('Only the two exact declared public XQ artifacts are accepted')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): digest.update(b)
    if digest.hexdigest() != want: raise RuntimeError('Checkpoint SHA256 mismatch before deserialization')
    defaults, reviewed = _globals()
    with zipfile.ZipFile(path) as z:
        entries = [x for x in z.namelist() if x.endswith('/data.pkl')]
        if len(entries) != 1: raise RuntimeError('Unexpected checkpoint pickle layout')
        if z.getinfo(entries[0]).file_size > 16*1024*1024:
            raise RuntimeError('Unexpectedly large checkpoint metadata')
        payload = z.read(entries[0])
    if len(payload) > 16*1024*1024: raise RuntimeError('Unexpectedly large checkpoint metadata')
    if not payload.startswith(b'\x80\x02'):
        raise RuntimeError('Only audited checkpoint pickle protocol2 is accepted')
    protocol_count = 0
    # Fixed protocol2 artifacts do not need extension registry or dynamic globals.
    for op,arg,pos in pickletools.genops(payload):
        if op.name == 'PROTO':
            protocol_count += 1
            if arg != 2 or pos != 0 or protocol_count != 1:
                raise RuntimeError('Only one initial protocol2 declaration is accepted')
        if op.name in ('EXT1','EXT2','EXT4','STACK_GLOBAL','INST','OBJ'):
            raise RuntimeError('Unreviewed pickle opcode: ' + op.name)
        if op.name=='GLOBAL' and '.'.join(arg.split(' ')) not in {**defaults,**reviewed}:
            raise RuntimeError('Unreviewed metadata global: ' + arg)
    unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - reviewed.keys()
    if unknown: raise RuntimeError('Unreviewed checkpoint globals: ' + repr(sorted(unknown)))
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False, pickle_module=_module())
