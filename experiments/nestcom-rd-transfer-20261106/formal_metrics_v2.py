"""Isolate unchanged metric definitions from RDVQ codec import side effects."""
import hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from runtime import R,definitions
from formal_codec_v1 import FiveMetrics as OriginalFiveMetrics

class FiveMetrics(OriginalFiveMetrics):
    def __init__(self,spec):
        ns=dict(Path=Path,hashlib=hashlib,importlib=__import__('importlib'),sys=sys,np=np,torch=torch,F=F)
        root=R/'experiments/rdvq-prefix-20261104'
        # Exact original helper/class bodies, with their executable source hashes
        # checked by the registered worker before construction. No codec imports.
        definitions(root/'training_core.py',{'verified'},ns)
        definitions(root/'metrics_v2.py',{'module','MetricSuite'},ns)
        self.base=ns['MetricSuite'](spec)
        with self.base.met.no_network():self.clip=self.base.met.load_clip(spec['clip'],torch.device('cuda:0'))
        self.clipcache={}
