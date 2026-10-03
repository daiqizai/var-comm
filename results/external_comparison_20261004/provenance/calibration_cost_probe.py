"""Time paid physical calibration only; no model updates or model selection."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--launch',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Use a fresh timing output')
    sys.path.insert(0,str(args.runtime.resolve()))
    import torch
    from swin_train import available_gpu,configure_runtime,model_hash
    from swin_data import RGBPopulation,sha,read
    from swin_model import build_official
    from swin_protocol import configure_phy,standard_noise
    from swin_replay import physical_batch
    command=read(args.launch)['cmd']
    value=lambda k:command[command.index('--'+k)+1]
    config=read(value('config'))
    available_gpu();configure_runtime();configure_phy(value('root'))
    population=RGBPopulation(value('image-cache'),value('reference-registration'),'calibration')
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    model=build_official(value('vendor')).eval()
    initial=model_hash(model)
    qualification=Path(value('output'))/'qualification.json'
    if initial!=read(qualification)['initial_model_sha256']:raise RuntimeError('Initial model differs')
    timings=[]
    with torch.no_grad():
        for N,snr in [(1024,1),(1024,13),(2048,1),(2048,13)]:
            torch.cuda.synchronize();started=time.perf_counter();accepted=0
            for start in range(0,100,16):
                indices=list(range(start,min(start+16,100)))
                noise=np.stack([standard_noise(population.ids[i],4101,N,snr) for i in indices])
                image=population.batch(indices,'cuda:0')
                reconstructed,ledger=physical_batch(model,image,N,snr,noise)
                # Include native per-frame MSE reduction in the timed boundary.
                mse=(reconstructed-image).square().flatten(1).mean(1).cpu().tolist()
                if len(mse)!=len(indices):raise RuntimeError('Calibration batch differs')
                accepted+=sum(r['header_accepted'] for r in ledger)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            timings.append(dict(N=N,snr=snr,frames=100,seconds=seconds,header_accepted=accepted))
    if model_hash(model)!=initial:raise RuntimeError('Timing probe changed model weights')
    result=dict(status='CALIBRATION_COST_PROBE_COMPLETE',scientific_result=False,
        formal_training_updates=0,model_selection=False,first100_original_calibration_sources=True,
        numerical_mode='registered FP32 deterministic',timings=timings,
        initial_model_sha256=initial,source_ids=population.ids[:100],
        input_bindings={**population.bindings,str(qualification):sha(qualification),
            str(Path(value('config'))):sha(value('config')),str(Path(__file__).resolve()):sha(__file__)})
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(timings),flush=True)


if __name__=='__main__':main()
