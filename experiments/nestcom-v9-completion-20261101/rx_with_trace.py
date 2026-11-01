"""Unchanged independent RX with observational CDF hashes, no TX inputs."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,asha,digest
import grid_codec, original_receiver
original_factory=grid_codec.codec
out=Path(read(sys.argv[1])['output'])
def traced_codec(native):
    c=original_factory(native);cdf=c._cdf;decode=c.decode;trace=[]
    def observed_cdf(family,provider,scale):
        value=cdf(family,provider,scale)
        trace.append(dict(scale=scale,cdf_sha256=asha(value)))
        return value
    def observed_decode(family,payload,m,K=0):
        trace.clear();value=decode(family,payload,m,K)
        key=digest(dict(family=family,m=m,K=K,payload_sha256=asha(payload)))
        save(out/'cdf_traces'/(key+'.json'),dict(family=family,m=m,K=K,payload_sha256=asha(payload),cdf_trace=list(trace),received_tokens_sha256=asha(value['received_tokens']),canonical=value['canonical'],zero_extension_reads=value.get('zero_extension_reads'),receiver_truth_used=value['receiver_truth_used'],transmitter_probability_table_used=value['transmitter_probability_table_used']))
        return value
    c._cdf=observed_cdf;c.decode=observed_decode
    return c
grid_codec.codec=traced_codec
original_receiver.L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/nestcom_v9_completion_20261101_v1')
original_receiver.main(*sys.argv[1:])
