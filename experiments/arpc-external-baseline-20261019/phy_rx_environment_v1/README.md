# Explicit RX environment correction after zero decoder calls

The original24 TX observations and source bindings remain immutable. Original RX
stopped before Runtime construction at a stdlib csv import; no header/body call,
ledger or image exists. CPY's actual stdlib is frozen_python_base_v1/lib/python3.11,
and its installed Torch/Sionna namespace uses the exact historical ldpc_environment
site-packages subtree. The original boundary omitted these two environment roots.

This revision only adds those two specific code roots and redirects fresh RX output
to rx_environment_v1. The original boundary implementation, packets, models, PHY
math and frame budget are unchanged. Source archives, source truth, TX, score and
model-output roots remain disallowed. No original file or failure is overwritten.

Four synthetic adapter tests and one CPU constructor-only admission run precede
recovery. The constructor probe forbids every transmit/receive/encode/decode method,
uses the production boundary, has no input waveforms/source/model weights, and
records zero packet calls. Successful original TX is inherited. Only24 previously
unattempted RX decodes may then execute; post-exit audit verifies the existing plan.
Unknown failures remain fatal and preserved; there is no automatic retry.
