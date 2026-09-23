import json
from pathlib import Path
import torch
from latent_enhancement.runtime import configure,digest,write_json,require_available,verify_snapshot
from short_prefix.common import Safety
EXP=Path(__file__).resolve().parents[2]
ROOT=EXP.parents[1]
OUT=ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
CONFIG=EXP/'protocol.json'
def read(p):return json.loads(Path(p).read_text())
def config():return read(CONFIG)
def configure_runtime():
    configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True

def bindings(files):return {str(Path(p).resolve()):digest(p) for p in files}

def register(p,record):
    if p.exists():
        if read(p)!=record:raise RuntimeError('run identity mismatch: '+str(p))
    else:write_json(p,record)
