"""Frozen LPIPS-VGG loss, local checked official weights; no network downloads."""
import json
from unittest.mock import patch
import torch
from torchvision import models
from runtime import OLD,sha
def load():
    import lpips
    spec=json.loads((OLD/'metric_assets.json').read_text());vgg=spec['dists']['vgg16_weights'];lin=spec['lpips_vgg']
    for s in [vgg,lin]:assert sha(s['path'])==s['sha256']
    constructor=models.vgg16
    def local(*args,**kwargs):
        m=constructor(weights=None);m.load_state_dict(torch.load(vgg['path'],map_location='cpu',weights_only=True));return m
    with patch.object(models,'vgg16',local):m=lpips.LPIPS(net='vgg',model_path=lin['path'],verbose=False)
    return m.cuda().float().eval().requires_grad_(False)
