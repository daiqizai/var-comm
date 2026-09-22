from __future__ import annotations
import argparse,json,time,copy,hashlib
from pathlib import Path
import torch
from latent_enhancement.runtime import image_losses,model_paths,perceptual_model,save_torch,settings
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_b.model import build_arms,latent_errors,render_received
from var_comm.next_scale_prior import load_models

PROJECT=Path(__file__).resolve().parents[5];VAR=PROJECT;ROOT=VAR/'outputs/VAR-LATENT-ENHANCEMENT-20260917';EXP=VAR/'experiments/var-latent-enhancement-20260917';CONFIG=EXP/'mechanisms/predictor_innovation_config.json'
def read(p):return json.loads(Path(p).read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def calibration_record(control,candidate,predictor,decoder,perceptual,calibration,scale,device,step):
 """Evaluate both arms on one fixed calibration slice without updating either arm."""
 control.eval(); candidate.eval(); predictor.eval()
 idx=torch.arange(min(16,len(calibration)),device=device); sni=torch.zeros(len(idx),dtype=torch.long); noi=torch.zeros(len(idx),dtype=torch.long)
 batch=calibration.batch(idx,sni,noi,[4101]*len(idx),device); snr=batch['snr_db']
 with torch.no_grad():
  canonical=torch.tensor([[1.,1.,8.]],device=device).expand(len(idx),3)
  ptx=predictor.receiver(None,batch['Fb_TX'],snr,canonical); prx=predictor.receiver(None,batch['Fb_RX'],snr,canonical)
  values=[]
  for name,model,target_base in (("control",control,batch['Fb_TX']),("candidate",candidate,ptx)):
   latent,_=model.receive_training_sample(batch['F'],target_base,prx,snr,batch['rx_status'],batch['standard_noise'])
   mse=(latent-batch['F']).square().mean().item(); values.append({'arm':name,'latent_mse':float(mse),'frames':len(idx)})
 return {'step':int(step),'algorithm':'matched_tx_conditions_shared_predictor_rx_start','rows':values}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'followup/predictor_innovation_v2'));p.add_argument('--resume');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cfg=read(CONFIG);device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);del var;torch.cuda.empty_cache();scale=scale_statistics(device);recipe=settings()['stage_B'];train=MatchedPopulation('train');calibration=MatchedPopulation('calibration');
 selected=read(ROOT/'stage_B_v1/training/selected_enhancement1024.json');
 if digest(selected['checkpoint'])!=selected['checkpoint_sha256']: raise RuntimeError('selected enhancement1024 checkpoint changed')
 base=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True);arms=build_arms(train.shape,scale,recipe).to(device);arms.load_state_dict(base['arms'],strict=True);control=arms['enhancement1024'];candidate=build_arms(train.shape,scale,recipe).to(device)['enhancement1024'];candidate.load_state_dict({k.split('enhancement1024.',1)[1]:v for k,v in base['arms'].items() if k.startswith('enhancement1024.')},strict=True);predictor_arms=build_arms(train.shape,scale,recipe).to(device);predsel=read(ROOT/'stage_B_v1/training/selected_receiver_only_refiner.json');
 if digest(predsel['checkpoint'])!=predsel['checkpoint_sha256']: raise RuntimeError('selected predictor checkpoint changed')
 predck=torch.load(predsel['checkpoint'],map_location='cpu',weights_only=True);predictor_arms.load_state_dict(predck['arms'],strict=True);predictor=predictor_arms['receiver_only_refiner'];predictor.eval().requires_grad_(False);decoder=load_decoder(vae,device);perceptual=perceptual_model(device);optc=torch.optim.AdamW(control.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);optx=torch.optim.AdamW(candidate.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);
 # The two arms start from independent optimizer copies.  A shared state dict
 # aliases Adam moments and makes the control update contaminate the innovation arm.
 optc.load_state_dict(copy.deepcopy(base['optimizers']['enhancement1024']));optx.load_state_dict(copy.deepcopy(base['optimizers']['enhancement1024']));order=torch.randperm(len(train));
 parent_step=int(base.get('step',40000)); incremental_step=0; calibration_rows=[]
 if a.resume:
  resume=torch.load(a.resume,map_location='cpu',weights_only=True); control.load_state_dict(resume['control'],strict=True); candidate.load_state_dict(resume['candidate'],strict=True); optc.load_state_dict(copy.deepcopy(resume['opt_control'])); optx.load_state_dict(copy.deepcopy(resume['opt_candidate'])); incremental_step=int(resume['step']);
 status={'status':'PREDICTOR_INNOVATION_V2_TRAINING','algorithm':'matched_tx_conditions_shared_predictor_rx_start','parent_step':parent_step,'incremental_step':incremental_step,'total_step':parent_step+incremental_step,'updates':cfg['updates'],'predictor_frozen':True,'predictor_condition':'canonical_status=[1,1,8] for TX and RX','control_checkpoint_sha256':selected['checkpoint_sha256'],'predictor_checkpoint_sha256':predsel['checkpoint_sha256'],'resume_from':a.resume or ''};(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 for step in range(incremental_step+1,cfg['updates']['maximum']+1):
  idx=order[((step-1)*16)%len(train):((step-1)*16)%len(train)+16];
  if len(idx)<16:idx=order[:16]
  sni=torch.randint(len(train.snrs),(16,));noi=torch.randint(len(train.seeds),(16,));batch=train.batch(idx,sni,noi,[recipe['channel_seed']+step]*16,device);snr=batch['snr_db'];
  with torch.no_grad():
   txstatus=torch.tensor([[1.,1.,8.]],device=device).expand(16,3);ptx=predictor.receiver(None,batch['Fb_TX'],snr,txstatus);prx=predictor.receiver(None,batch['Fb_RX'],snr,txstatus)
  for model,opt,innovation in ((control,optc,False),(candidate,optx,True)):
   opt.zero_grad(set_to_none=True);target_base=ptx if innovation else batch['Fb_TX'];rx_base=prx;continuous_arg=batch['F'];latent,_=model.receive_training_sample(continuous_arg,target_base,rx_base,snr,batch['rx_status'],batch['standard_noise']);pred=render_received(decoder,latent,batch['rx_status']);mse,lp=image_losses(pred,batch['target'],perceptual);aux=latent_errors(latent,batch['F'],scale,batch['rx_status']);loss=(mse+.1*lp+.01*aux).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step%cfg['updates']['full_calibration_interval']==0:
   calibration_rows.append(calibration_record(control,candidate,predictor,decoder,perceptual,calibration,scale,device,step));(out/'calibration.json').write_text(json.dumps(calibration_rows,indent=2)+'\n')
  if step%cfg['updates']['checkpoint_interval']==0:
   status.update({'step':step,'incremental_step':step,'total_step':parent_step+step,'last_loss':float(loss),'timestamp':time.time()});(out/'status.json').write_text(json.dumps(status,indent=2)+'\n');save_torch(out/f'step_{step:05d}.pt',{'control':control.state_dict(),'candidate':candidate.state_dict(),'opt_control':optc.state_dict(),'opt_candidate':optx.state_dict(),'predictor_state':predictor.state_dict(),'step':step,'parent_step':parent_step,'incremental_step':step,'total_step':parent_step+step})
 selected_step=max(calibration_rows[-1]['step'] if calibration_rows else cfg['updates']['maximum'],1); selected_path=out/f'step_{selected_step:05d}.pt';
 if selected_path.is_file(): (out/'selected.json').write_text(json.dumps({'checkpoint':str(selected_path),'checkpoint_sha256':digest(selected_path),'algorithm':status['algorithm'],'parent_step':parent_step,'incremental_step':selected_step,'total_step':parent_step+selected_step},indent=2)+'\n')
 (out/'completion.json').write_text(json.dumps({'status':'PREDICTOR_INNOVATION_V2_COMPLETE','algorithm':status['algorithm'],'parent_step':parent_step,'incremental_step':selected_step,'total_step':parent_step+selected_step,'calibration_calls':len(calibration_rows),'predictor_condition':'canonical_status=[1,1,8] shared at TX/RX','new_holdout_used':False},indent=2)+'\n'); print('predictor innovation v2 complete')
if __name__=='__main__':main()
