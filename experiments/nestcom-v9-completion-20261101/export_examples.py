"""Render two preregistered examples from closed native arrays; zero inference."""
from pathlib import Path
import json,csv,ast,sys
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,pin,checked,asha
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');D=R/'results/nestcom_v9_completion_20261101_v1'
def rows(p):return list(csv.DictReader(p.open()))
def main():
    selection=read(D/'example_selection.json');out=D/'examples';out.mkdir(exist_ok=False)
    old=R/'results/nestcom_perceptual_followup_20261030_v2/closed_evidence_v1/e1'
    metrics=rows(old/'dists_per_frame.csv');inventory={v['dists_pair_key']:v for v in rows(old/'native_output_inventory.csv')}
    sources=read(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json');sources={s['source_id']:s for s in sources};mapping=[]
    for choice in selection['rows']:
        sid=choice['source_id'];s=sources[sid]
        fs=[f for f in metrics if f['source_id']==sid and f['N']=='1024'and f['snr_db']=='10'and f['channel_repeat']=='0'and f['method']in ['final_NeST','WHOLE_final_receiver']];assert len(fs)==2
        archive=s.get('archive',s.get('source'));assert archive is not None,list(s)
        with np.load(checked(archive),allow_pickle=False)as z:
            x=z['pixels'].copy()
        assert x.dtype==np.uint8 and x.shape==(3,256,256)
        assert asha(x.astype(np.float32)/255)==fs[0]['reference_sha256']==fs[1]['reference_sha256']
        name=choice['selection']+'_source.png';Image.fromarray(x.transpose(1,2,0)).save(out/name);mapping.append(dict(selection=choice['selection'],method='Source',source_id=sid,source_index=choice['source_index'],display=pin(out/name),reference=archive))
        for f in fs:
            inv=inventory[f['dists_pair_key']];image=ast.literal_eval(inv['image'])if inv['image']else ast.literal_eval(inv['archive'])
            with np.load(checked(image),allow_pickle=False)as z:
                y=z[inv['image_key']or'image'].copy()
                if inv['image_slot']:y=y[int(inv['image_slot'])]
            assert y.dtype==np.float32 and y.shape==(3,256,256)and asha(y)==f['pixel_sha256']
            method='Token'if f['method']=='final_NeST'else'WHOLE';name=choice['selection']+'_'+method+'.png'
            Image.fromarray(np.rint(np.clip(y,0,1)*255).astype(np.uint8).transpose(1,2,0)).save(out/name)
            mapping.append(dict(selection=choice['selection'],method=method,source_id=sid,source_index=f['source_index'],display=pin(out/name),native=image,pixel_sha256=f['pixel_sha256'],dists=float(f['dists']),dreamsim=float(f['dreamsim']),noise_seed=f['noise_seed'],repeat=0,receiver_status=f['receiver_status']))
    save(out/'mapping.json',mapping);save(out/'completion.json',dict(status='COMPLETE',selected_sources=2,images=6,new_reconstruction=0,new_scoring=0,selection=pin(D/'example_selection.json')));print(mapping)
if __name__=='__main__':main()
