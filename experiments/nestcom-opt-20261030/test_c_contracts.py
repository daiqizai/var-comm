import torch,json
from c_model import Corrector,missing_loss
torch.set_num_threads(2);torch.manual_seed(2026103001)
m=Corrector();m.train();count=sum(x.numel()for x in m.parameters());proof=[]
for s,h,Kv in [(7,10,25),(8,13,42)]:
 B,L=2,h*h;coarse=torch.randn(B,L,32);base=torch.randn(B,L,4096);emb=torch.randn(B,L,32);K=torch.full((B,),Kv);mask=torch.arange(L)[None]<K[:,None]
 y=m(coarse,base,emb,mask,s,K);assert torch.equal(y,base)
 with torch.no_grad():m.head.weight.normal_(0,.001)
 a=m(coarse,base,emb,mask,s,K,no_content=True);b=m(coarse,base,emb*7,mask,s,K,no_content=True);assert torch.equal(a,b)
 a=m(coarse,base,emb,mask,s,K);assert torch.equal(a[mask],base[mask]);target=torch.randint(4096,(B,L));loss=missing_loss(base,a,target,mask);loss.backward()
 assert torch.isfinite(loss)and all(torch.isfinite(p.grad).all()for p in m.parameters()if p.grad is not None)
 with torch.no_grad():m.head.weight.zero_();m.head.bias.zero_()
 m.zero_grad();proof.append(dict(m=s,zero_residual_exact=True,known_logits_fixed=True,no_content_invariant=True,finite_gradients=True))
print(json.dumps(dict(status='CPU_ENGINEERING_PASS_NOT_IMAGE_EVIDENCE',parameters=count,proof=proof)))
