"""Narrow wire rejection boundary; all model/CDF/asset errors stay fatal."""
import numpy as np
from common import asha
from c_arithmetic import canonical


class SourceRejected(Exception):
    def __init__(self,reason,tokens,traces):
        super().__init__(reason)
        self.reason=reason;self.tokens=list(tokens);self.traces=list(traces)


def decode_prefix(n,Encoder,Decoder,cdf,bits,prefix,predicted_class,work):
    # No original image, source token, TX CDF or intended class enters here.
    n.start(predicted_class)
    dec=Decoder(bits);enc=Encoder();tokens=[];traces=[]
    for i in range(prefix):
        logits=n.logits(i)
        work['prior_forwards']+=1;work['prior_branch_forwards']+=2
        table=n.cdf(logits,cdf);traces.append(asha(table))
        try:
            v=dec.decode(table)
        except ValueError as e:
            if str(e)!='Arithmetic terminal lookahead exceeded':raise
            raise SourceRejected('ARITHMETIC_TERMINAL_LOOKAHEAD_EXCEEDED',tokens,traces)from e
        enc.encode(v,table);tokens.append(int(v[0]));n.previous=tokens[-1]
    try:
        canonical(dec,enc,bits)
    except AssertionError as e:
        reasons={'Noncanonical arithmetic length/termination':'NONCANONICAL_ARITHMETIC_LENGTH_OR_TERMINATION',
            'Arithmetic terminal lookahead mismatch':'ARITHMETIC_TERMINAL_LOOKAHEAD_MISMATCH'}
        if str(e)not in reasons:raise
        raise SourceRejected(reasons[str(e)],tokens,traces)from e
    return tokens,traces


def realize(n,Encoder,Decoder,cdf,bits,prefix,predicted_class,seed):
    work=dict(prior_forwards=0,prior_branch_forwards=0,decoder_forwards=0)
    try:
        tokens,traces=decode_prefix(n,Encoder,Decoder,cdf,bits,prefix,predicted_class,work)
    except SourceRejected as e:
        # A protocol failure is a completed received outcome, not an engineering
        # retry. No tail generation or neural image decoder runs after rejection.
        return dict(status='FIXED_GRAY_SOURCE_REJECT',reason=e.reason,
            image=np.full((3,256,256),.5,np.float32),
            received_tokens=np.array(e.tokens,np.int64),completed_tokens=np.empty(0,np.int64),
            cdf_hashes=e.traces,**work)
    received=np.array(tokens,np.int64)
    with n.torch.random.fork_rng(devices=[0]):
        n.torch.manual_seed(seed)
        for i in range(prefix,256):
            logits=n.logits(i);v=n.tail(logits);tokens.append(int(v.item()));n.previous=tokens[-1]
            work['prior_forwards']+=1;work['prior_branch_forwards']+=2
    tokens=np.array(tokens,np.int64);image=n.render(tokens);work['decoder_forwards']=1
    return dict(status='ACCEPTED_SOURCE',reason=None,image=image,received_tokens=received,
        completed_tokens=tokens,cdf_hashes=traces,**work)
