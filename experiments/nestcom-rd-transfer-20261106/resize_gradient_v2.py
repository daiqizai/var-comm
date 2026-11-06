"""Exact native interpolation forward with a deterministic separable adjoint.

PyTorch CUDA area backward is unavailable under strict determinism. All three
linear resizing operators here have fixed one-dimensional interpolation
matrices. Forward is unchanged; backward multiplies by their transposes.
"""
import torch
from torch.nn import functional as F
CACHE={}
def native(x,size,mode):
    return F.interpolate(x,size=size,mode=mode,**({'align_corners':False} if mode=='bicubic' else {}))
def matrix(n,m,mode,device,dtype):
    key=(n,m,mode,str(device),dtype)
    if key not in CACHE:
        basis=torch.eye(n,dtype=torch.float64).reshape(n,1,n,1)
        weights=native(basis,(m,1),mode)[:,0,:,0].T.contiguous()
        CACHE[key]=weights.to(device=device,dtype=dtype)
    return CACHE[key]
class Resize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,h,w,mode):
        wh=matrix(x.shape[-2],h,mode,x.device,x.dtype);ww=matrix(x.shape[-1],w,mode,x.device,x.dtype)
        ctx.save_for_backward(wh,ww)
        return native(x,(h,w),mode)
    @staticmethod
    def backward(ctx,grad):
        wh,ww=ctx.saved_tensors
        return torch.matmul(torch.matmul(wh.T,grad),ww),None,None,None
def resize(x,size,mode):
    return Resize.apply(x,size[0],size[1],mode)
def check():
    torch.manual_seed(2026110601);rows=[]
    for mode in ['area','bicubic','nearest']:
        for n,m in [(16,13),(13,16),(3,6)]:
            x=torch.randn(1,2,n,n,dtype=torch.float64,requires_grad=True);g=torch.randn(1,2,m,m,dtype=torch.float64)
            a=native(x,(m,m),mode);b=resize(x,(m,m),mode);assert torch.equal(a,b)
            ga=torch.autograd.grad(a,x,g)[0];gb=torch.autograd.grad(b,x,g)[0]
            error=float((ga-gb).abs().max());assert error<1e-12,(mode,n,m,error)
            rows.append(dict(mode=mode,input=n,output=m,max_derivative_error=error))
    return rows
