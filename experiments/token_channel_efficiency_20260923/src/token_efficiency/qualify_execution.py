"""Real-weight acceptance for shared quality/timing; no formal quality metrics.

Run only with authorized GPU0 free, before launching the digital quality grid.
Use calibration sources only. Budget models here are disposable seeded probes;
selected trained models must be replay-verified again at their later evaluation.
"""
from pathlib import Path
import numpy as np,torch
from latent_enhancement.runtime import require_available,model_paths,write_json,digest,verify_snapshot
from latent_enhancement_b.common import load_decoder,scale_statistics
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.prefix_training_data import read_image_population
from short_prefix.common import identity_files
from .common import ROOT,OUT,CONFIG,read,register,configure_runtime,Safety
from .models import BudgetContinuous
from .execution import Cell,execute,receive,transmit,apply_channel,load_selected_budget,prepare_digital,transmit_prepared
from .codec import cumulative

def main():
    configure_runtime();require_available();device=torch.device('cuda:0');safety=Safety()
    out=OUT/'execution_qualification_v1';out.mkdir(parents=True,exist_ok=True)
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device)
    images,labels,ids,image_bindings=read_image_population('calibration')
    local=Path(__file__).parent
    files=[CONFIG,*[local/name for name in ('qualify_execution.py','execution.py','codec.py','phy.py','models.py','common.py')]]
    identity={'bindings':identity_files(files),'data':image_bindings[:2],'models':model_paths(),'decoder_sha256':state_sha256(decoder),'calibration_only':True,'synthetic':False,'trained_model_quality_claim':False}
    register(out/'registration.json',identity);frozen={k:state_sha256(m) for k,m in [('vae',vae),('var',var),('Dc',decoder)]};rows=[]
    with torch.no_grad():
        for N in (2048,3060):
            torch.manual_seed(2026092304);model=BudgetContinuous(scale.cpu(),N).to(device).eval()
            for i in range(2):
                if safety.check():raise RuntimeError('sustained thermal condition')
                record={'pixels':images[i].numpy(),'image_id':ids[i],'class_index':int(labels[i])};cell=Cell('continuous',N)
                for snr in (1,13):
                    image,info=execute(record,cell,snr,4101,vae,var,decoder,device,model)
                    replay,_=receive(info['observation'],snr,cell,vae,var,decoder,device,model)
                    np.testing.assert_array_equal(image,replay)
                    F=vae.quant_conv(vae.encoder(images[i:i+1].to(device).float()/127.5-1))
                    from var_comm.study import seeded_noise
                    noise=torch.as_tensor(seeded_noise(f'VAR-CONTINUOUS-{N}|'+str(ids[i]),4101,(N,2)),device=device,dtype=torch.float32)[None]
                    latent,wave=model(F,torch.tensor([snr],device=device),noise)
                    np.testing.assert_allclose(wave[0].cpu().numpy(),info['waveform'],atol=0,rtol=0)
                    np.testing.assert_allclose(decoder(latent)[0].cpu().numpy(),image,atol=2e-5,rtol=1e-5)
                    rows.append({'cell':cell.name,'source_index':i,'snr':snr,'N':N,'E':info['ledger']['E'],'online_training_match':True,'replay_exact':True,'synthetic':False})
            del model
        # Probe both newly introduced endpoint modes at actual waveform lengths.
        record={'pixels':images[0].numpy(),'image_id':ids[0],'class_index':int(labels[0])}
        F=vae.quant_conv(vae.encoder(images[:1].to(device).float()/127.5-1));tokens=vae.quantize.f_to_idxBl_or_fhat(F,to_fhat=False);source=[x[0].cpu().numpy() for x in tokens]
        for N in (2048,3060,4084):
            for family in ('raw','arithmetic'):
                for mcs in ('QPSK','16QAM'):
                    for mode in (6,10):
                        if safety.check():raise RuntimeError('thermal pause')
                        cell=Cell(family,N,mode,mcs)
                        from .phy import dimensions
                        _,_,slots=dimensions(N,family,mcs)
                        if family=='raw' and 12*sum(len(x) for x in source[:mode])+22>slots:
                            rows.append({'cell':cell.name,'status':'UNENCODABLE_LENGTH_CONSTRAINT','synthetic':False});continue
                        wave,ledger=transmit(record['pixels'],record['class_index'],cell,vae,var,device)
                        prepared=prepare_digital(record['pixels'],record['class_index'],family,vae,var,device)
                        cached_wave,cached_ledger=transmit_prepared(prepared,record['class_index'],cell)
                        np.testing.assert_array_equal(wave,cached_wave);assert ledger==cached_ledger
                        clean,event=receive(wave,19,cell,vae,var,decoder,device)
                        if ledger['source_overflow_erasure']:
                            assert not event['header_ok'];np.testing.assert_array_equal(clean,np.full_like(clean,.5))
                        else:
                            assert event['header_ok'] and event['body_crc_ok'] and event['source_complete'] and event['decoded_label']==record['class_index'] and event['decoded_mode']==ledger['m_actual']
                            from latent_enhancement.latent import complete_latent
                            actual=ledger['m_actual'];latent=cumulative(vae,source,10) if actual==10 else complete_latent(vae,var,source[:actual],record['class_index'],device)
                            np.testing.assert_allclose(clean,decoder(latent)[0].cpu().numpy(),atol=2e-5,rtol=1e-5)
                        noisy,info=execute(record,cell,1,4101,vae,var,decoder,device)
                        replay,_=receive(info['observation'],1,cell,vae,var,decoder,device);np.testing.assert_array_equal(noisy,replay)
                        rows.append({'cell':cell.name,'N':N,'E':ledger['E'],'source_index':0,'noiseless_replay':True,'actual_low_snr_replay':True,'body_crc_ok_at_1dB':bool(info['rx']['body_crc_ok']),'synthetic':False})
    for name,model in [('vae',vae),('var',var),('Dc',decoder)]:
        if state_sha256(model)!=frozen[name] or any(p.grad is not None for p in model.parameters()):raise RuntimeError('frozen weight/gradient changed')
    verify_snapshot(identity['bindings']);write_json(out/'acceptance.json',{'status':'REAL_WEIGHT_SHARED_EXECUTION_PASS','registration_sha256':digest(out/'registration.json'),'checks':rows,'frozen_model_states':frozen,'synthetic':False,'formal_quality_metrics_generated':False,'selected_model_requalification_required':True})
    print('REAL_EXECUTION_ACCEPTANCE_PASS',len(rows),flush=True)
if __name__=='__main__':main()
