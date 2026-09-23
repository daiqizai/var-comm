"""Read real author checkpoints with GPU visibility disabled before importing torch."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import torch
from external_positioning.author_models import DiffComAuthor
from tools.replay_author_reference_timing import runtime, frozen_sha, EXP, code_files
from latent_enhancement.runtime import digest, write_json


def main():
    record=runtime()
    record.update(synthetic=False,GPU_forward='NOT_RUN',CPU_real_strict_loading=[],CUDA_VISIBLE_DEVICES='')
    assert torch.cuda.device_count()==0
    checkpoint=EXP/'checkpoints/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model'
    weights=torch.load(checkpoint,map_location='cpu')
    assert all(v.device.type=='cpu' and torch.isfinite(v).all() for v in weights.values())
    record['Swin']={'status':'CPU_CHECKPOINT_DESERIALIZATION_PASS_STRICT_MODEL_CPU_CONSTRUCTOR_UNSUPPORTED',
                    'checkpoint_sha256':digest(checkpoint),'state_entries':len(weights),
                    'reason':'upstream encoder.py update_mask calls attn_mask.cuda() even with CPU device; real native parity awaits serial GPU stage'}
    del weights
    for rate in (2,4,6):
        checkpoint=EXP/'checkpoints'/f'ADJSCC_C={rate}.pth.tar'
        model=DiffComAuthor(checkpoint,rate,device='cpu')
        record['CPU_real_strict_loading'].append({'family':'adjscc','rate':rate,'checkpoint_sha256':digest(checkpoint),'frozen_state_sha256':frozen_sha(model),'parameters':model.parameters})
        del model
    record.update(status='ADJSCC_REAL_CPU_STRICT_LOAD_AND_SWIN_CHECKPOINT_READ_PASS_GPU_NOT_RUN',CUDA_initialized=torch.cuda.is_initialized())
    assert not record['CUDA_initialized']
    record['code_bindings']={str(p.relative_to(Path.cwd())):digest(p) for p in [*code_files(),Path(__file__).resolve()]}
    write_json('results/token_channel_efficiency_20260923/author_native_timing_queue_v1/cpu_loading.json',record)
    print('MASKED_CPU_SCOPE_PASS; CUDA not initialized; GPU forward NOT_RUN')

if __name__=='__main__':main()
