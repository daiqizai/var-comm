from pathlib import Path
import json,random
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from runtime import sha
def image(row,position=None):
    assert sha(row['path'])==row['sha256']
    with Image.open(row['path']) as im:im=im.convert('RGB')
    im=TF.resize(im,256,interpolation=TF.InterpolationMode.BICUBIC,antialias=True)
    w,h=im.size
    if position is None:im=TF.center_crop(im,[256,256])
    else:
        g=random.Random(f'2026110601:{position}:{row["source_id"]}')
        im=TF.crop(im,g.randrange(h-255),g.randrange(w-255),256,256)
        if g.random()<.5:im=TF.hflip(im)
    return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/127.5-1
class Corpus:
    def __init__(self,path):
        self.rows=json.loads(Path(path).read_text())['sources'];self.orders={}
    def get(self,position):
        epoch,j=divmod(position,len(self.rows))
        if epoch not in self.orders:
            order=list(range(len(self.rows)));random.Random(2026110601+epoch).shuffle(order);self.orders={epoch:order}
        row=self.rows[self.orders[epoch][j]]
        return image(row,position),row['source_id']
