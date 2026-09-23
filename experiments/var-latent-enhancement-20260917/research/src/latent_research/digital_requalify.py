"""Requalify only same-Dc N4084 digital candidates at the registered precision."""
import argparse,json,time
from pathlib import Path
import torch,yaml
from .train import ROOT,PROJECT
from latent_enhancement.runtime import configure,require_available,model_paths,write_json,digest,snapshot,verify_snapshot
from latent_enhancement_b.common import load_decoder
from latent_followup.digital_policy_matrix import load_source,evaluate_source,write_rows
from latent_enhancement_eval.runner import load_targets
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.progressive import split_prefix
from var_comm.quality import load_quality_models

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    configure();require_available();device=torch.device('cuda:0');paths=model_paths();vae,var=load_models(paths,device);decoder=load_decoder(vae,device)
    cfg=yaml.safe_load((PROJECT/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,_=load_quality_models(cfg,device)
    import latent_followup.digital_policy_matrix as shared
    identity={'source_snapshot':snapshot([__file__,shared.__file__]),'model_paths':paths,'decoder_state_sha256':state_sha256(decoder),
        'precision':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32},
        'N':4084,'E':8168,'modes':[7,8,9],'renderer':'Dc','roles':{'calibration':1000,'development':100},'new_holdout_used':False}
    reg=out/'registration.json'
    if reg.exists():
        if json.loads(reg.read_text())!=identity:raise RuntimeError('digital resume identity changed')
        verify_snapshot(identity['source_snapshot'])
    else:write_json(reg,identity)
    regsha=digest(reg);all_rows={};snrs=[1.,4.,7.,13.,19.]
    for role,count,seeds in [('calibration',1000,[4101,4102,4103]),('development',100,[2001,2002,2003])]:
        targets=load_targets() if role=='development' else None
        rows=[]
        for i in range(count):
            folder=out/role/f'{i:04d}';receipt=folder/'receipt.json';csvpath=folder/'per_frame.csv'
            if receipt.exists():
                r=json.loads(receipt.read_text())
                if r['registration_sha256']!=regsha or digest(csvpath)!=r['sha256']:raise RuntimeError('digital source cache changed')
                import csv
                rows.extend(csv.DictReader(csvpath.open()));continue
            source=load_source(i) if role=='calibration' else {'index':i,'image_id':targets[i]['target']['image_id'],
                'label':int(targets[i]['target']['class_index']),'source':split_prefix(targets[i]['tokens'],10),'source_rgb':targets[i]['pixels'].astype('float32')/255}
            with torch.no_grad():current=evaluate_source(source,vae,var,decoder,lp,device,4084,seeds,snrs,renderers=('Dc',),dino=dino if role=='development' else None)
            for r in current:r['population']=role
            if len(current)!=90:raise RuntimeError('digital candidate source incomplete')
            folder.mkdir(parents=True,exist_ok=True);write_rows(csvpath,current);write_json(receipt,{'source_index':i,'image_id':source['image_id'],'sha256':digest(csvpath),'registration_sha256':regsha})
            rows.extend(current);print('digital strict',role,i+1,flush=True)
        write_rows(out/f'{role}.csv',rows);all_rows[role]=len(rows)
    write_json(out/'completion.json',{'status':'N4084_DC_DIGITAL_STRICT_PRECISION_COMPLETE','rows':all_rows,
        'registration_sha256':regsha,'calibration_sha256':digest(out/'calibration.csv'),'development_sha256':digest(out/'development.csv'),'new_holdout_used':False})
if __name__=='__main__':main()
