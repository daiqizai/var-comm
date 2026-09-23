import hashlib,json,os,subprocess,time
from pathlib import Path
import torch
from latent_enhancement.runtime import configure,digest,write_json,require_available,model_paths,verify_snapshot
from latent_enhancement_b.common import CACHE,decoder_gate_path,bind_files
from .protocol import ROOT,EXP,CONFIG,config

OUT=ROOT/'outputs/SHORT-PREFIX-20260923'

def read(p):return json.loads(Path(p).read_text())

def configure_runtime():
    configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True

def identity_files(extra=()):
    old=ROOT/'experiments/var-latent-enhancement-20260917'
    files=[CONFIG,decoder_gate_path(),CACHE/'training_statistics.json',ROOT/'configs/next_scale_prior_diagnostic.yaml',ROOT/'configs/progressive_channel.yaml']
    files += [Path(__file__),Path(__file__).with_name('protocol.py'),Path(__file__).with_name('models.py')]
    for d in [ROOT/'src/var_comm',old/'src',old/'phase_b/src']:
        files += list(d.rglob('*.py'))
    return bind_files([*files,*extra])

def register(path,value):
    if path.exists():
        if read(path)!=value:raise RuntimeError(f'identity changed: {path}')
    else:write_json(path,value)

class Safety:
    def __init__(self):self.hot_count=0;self.last=None
    def check(self):
        require_available()
        fields=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=temperature.gpu,clocks_event_reasons.hw_thermal_slowdown','--format=csv,noheader,nounits'],text=True).strip().split(',')
        hot=int(fields[0])>=86 or fields[1].strip().lower()=='active'
        self.hot_count=self.hot_count+1 if hot else 0
        self.last={'temperature':int(fields[0]),'hardware_thermal_slowdown':fields[1].strip(),'consecutive_hot':self.hot_count,'time':time.time()}
        return self.hot_count>=3

def tensor_sha(x):return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()
