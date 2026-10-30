import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
assert not(D/'benchmark_tx_registration.json').exists()
proof=read(D/'benchmark_protocol.json');old=cr(proof['tx_cases']);safe=[{k:v for k,v in x.items()if k!='physical_cases'}for x in old]
allowed={'case_id','source_id','source_index','N','snr_db','archive','source_pixels_sha256','target','capacity_bits','q','rate','expected_payloads'};assert all(set(v)==allowed for v in safe)
p=L/'benchmark_inputs/tx_source_cases.json';save(p,safe);save(D/'benchmark_TX_input_projection.json',dict(original=proof['tx_cases'],TX_source_only=pin(p),fields=sorted(allowed),excludes_actual_RX_projection_frame_receipts_noise_counters=True,expected_payload_hashes='Known transmitter outputs used only for post-call equality checks; never input to selection',before_first_TX_benchmark=True))
print('TX_SOURCE_ONLY_INPUT_PREPARED')
