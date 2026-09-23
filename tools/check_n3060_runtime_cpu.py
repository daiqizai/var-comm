"""Real selected-state loading only; never perform GPU or image-quality evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import torch
from tools.n3060_reference_execution import Systems
from tools.replay_n3060_reference_timing import runtime_identity,ROOT
from latent_enhancement.runtime import configure,write_json,digest


def main():
    runtime=runtime_identity();configure()
    assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
    models=Systems('cpu');assert not torch.cuda.is_initialized()
    loaded={}
    for name,module in tuple(sys.modules.items()):
        if name.startswith(('var_comm','cadsd_jscc','wetok_comm','vector_control','_wetok_pinned','tools.n3060_reference')) and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve()
            assert path.is_relative_to(ROOT) and 'publish' not in path.relative_to(ROOT).parts
            loaded[name]={'path':str(path.relative_to(ROOT)),'sha256':digest(path)}
    write_json(ROOT/'results/token_channel_efficiency_20260923/n3060_timing_queue_v1/cpu_loading.json',
        {'status':'REAL_N3060_SELECTED_MODELS_CPU_STRICT_LOAD_AND_HISTORICAL_STATE_IDENTITY_PASS',
         'models':models.before,'selected':models.selected,'GPU_forward':'NOT_RUN','CUDA_initialized':False,
         'synthetic':False,'runtime':runtime,'loaded_root_modules':loaded,'check_tool_sha256':digest(__file__)})
    print('REAL_N3060_CPU_STRICT_LOADING_PASS_NO_CUDA')

if __name__=='__main__':main()
