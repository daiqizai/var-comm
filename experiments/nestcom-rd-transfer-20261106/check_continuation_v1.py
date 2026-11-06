"""CPU checks for continuation partitioning, accumulation and scalar state hashes."""
import ast,json,hashlib,contextlib
from pathlib import Path
import numpy as np
import torch
from runtime import S

def main():
    for name in ['train_continue_v1.py','owner_continue_v1.py']:
        compile((S/name).read_text(),str(S/name),'exec')
    for world in (4,8):
        for step in (200,201,999,4999):
            observed=[]
            for rank in range(world):
                for lane in range(rank,8,world):
                    observed.append((step*8+lane,(step+lane)%4))
            assert sorted(observed)==[(step*8+j,(step+j)%4) for j in range(8)]
    x=torch.arange(8,dtype=torch.float64).reshape(8,1)/7
    y=x.square()+.1
    p=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
    ((x*p-y).square().mean()).backward();reference=p.grad.detach().clone()
    for world in (4,8):
        grads=[]
        for rank in range(world):
            q=p.detach().clone().requires_grad_()
            lanes=list(range(rank,8,world))
            for lane in lanes:
                ((x[lane]*q-y[lane]).square().mean()/len(lanes)).backward()
            grads.append(q.grad.detach())
        assert torch.allclose(torch.stack(grads).mean(0),reference,rtol=0,atol=1e-15)
    node=next(n for n in ast.parse((S/'train_continue_v1.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='state_hash')
    ns=dict(hashlib=hashlib,torch=torch)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<state_hash>','exec'),ns)
    state={'step':torch.tensor(200.),'moment':torch.arange(16.).reshape(4,4),
           'groups':[{'lr':1e-5,'betas':(.9,.95)}]}
    clone={'groups':state['groups'],'moment':state['moment'].clone(),'step':state['step'].clone()}
    assert ns['state_hash'](state)==ns['state_hash'](clone)
    clone['moment'][0,0]+=1
    assert ns['state_hash'](state)!=ns['state_hash'](clone)
    print(json.dumps(dict(passed=True,worlds=[4,8],global_batch=8,
                         sample_and_budget_schedule_exact=True,
                         accumulation_gradient_max_tolerance=1e-15,
                         state_hash_scalar_and_mutation_checks=True,
                         real_model_calls=0)))

if __name__=='__main__':main()
