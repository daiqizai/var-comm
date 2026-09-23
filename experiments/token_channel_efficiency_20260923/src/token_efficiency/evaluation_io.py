"""Strict resumable real evaluation records and thermal-safe inference boundaries."""
import signal,time
from pathlib import Path
from latent_enhancement.runtime import digest,write_json,ResourceBusy,require_available
from .common import read
from .thermal_guard import sample,Window

class SafeEvaluation:
    def __init__(self):
        self.stop=False;self.window=Window();self.last=0.;self.hardware=None
        signal.signal(signal.SIGTERM,self.halt);signal.signal(signal.SIGINT,self.halt)
    def halt(self,*_):self.stop=True
    def check(self):
        if self.stop:raise ResourceBusy('evaluation checkpoint boundary requested')
        if time.monotonic()-self.last>=10:
            require_available();self.hardware=sample();hot,_=self.window.add(self.hardware);self.last=time.monotonic()
            if hot:raise ResourceBusy('sustained hardware/software thermal condition')

def load_cell(path,registration_sha,source_id,method):
    path=Path(path);seal=path.with_suffix('.seal.json')
    if not path.exists():
        if seal.exists():raise RuntimeError('cell seal without data')
        return None
    if not seal.exists():
        # Preserve an interrupted atomic write before recomputing its cell.
        backup=path.parent/'uncommitted';backup.mkdir(exist_ok=True)
        path.rename(backup/f'{path.stem}_{time.time_ns()}.json');return None
    receipt=read(seal)
    if receipt['sha256']!=digest(path):raise RuntimeError('cell hash changed')
    rec=read(path)
    if rec['registration_sha256']!=registration_sha or rec['source_id']!=source_id or rec['method']!=method:raise RuntimeError('cell resume scope changed')
    return rec

def save_cell(path,record):
    path=Path(path)
    if path.exists() or path.with_suffix('.seal.json').exists():raise RuntimeError('committed cell cannot be overwritten')
    write_json(path,record);write_json(path.with_suffix('.seal.json'),{'sha256':digest(path),'registration_sha256':record['registration_sha256']})

def save_preview(path,image):
    """Server-local reconstruction only. Never affects float quality or PHY."""
    import numpy as np
    from PIL import Image
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(np.clip(image.transpose(1,2,0),0,1)*255).astype(np.uint8)).save(path)
    return {str(path):digest(path)}
