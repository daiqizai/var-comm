# Fullcal1000 physical calibration, subordinate to A2b master budget

Only the six previously frozen candidate/SNR combinations run. All1000 sources,
original4101/4102/4103 noises, fullN1024=68+956 and nominalEs2 remain. At most18000
logical frames and18000 header+18000 body decoder calls. Originalcal1000 source
traversals must close first. No model reconstruction, quality score or bootstrap
is executed by this entry. The existing A2b absolute deadline is inherited.

Each source/MCS/actual prefix has one deterministic complete transmitted waveform;
exact waveform reuse can avoid repeat LDPC encoding. Different SNR/noise observations
are still distinct physical events and all18000 receivers are evaluated. The public
counter is image-independent in meaning and does not encode a free prompt/token.
The full96 paid-profile catalogue is unchanged from the closed engineering stage.

TX uses actual standalone payload length to fall back. No complete prefix fits:
use explicit null and full frame. Receiver accepts only actual observed header,
body, CRC, paid length/padding and actual source grammar. Source and TX roots are
blocked. Accepted but wrong information is retained and dispatched to subsequent
source RX; post-exit truth audit is descriptive only, never an error repair rule.

Every header/body call is durably reserved and completed in an append-only ledger.
Incomplete or failed calls stop the chain; no automatic replay. Fixed failure
images and exact cached RX-state reuse are handled only by the next independently
registered reconstruction/score stage. No cal100 pre-screen score is a final policy.
