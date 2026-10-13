# R1 checkpoint pin metadata compatibility

This isolated adapter repairs a zero-science registration stop. All 200 cal200
checkpoint pins match the frozen original1000 selection in path and SHA; the
calibration manifest additionally records the correct file byte count.

`pin_adapter.py` accepts exactly `path` and `sha256`, optionally `bytes`, for
checkpoint pins. Each supplied byte count must be an integer (not bool) and must
match the real file. Both file SHA values are checked. Every other selected source
field remains exactly equal. The sole original PHY expression replacement has an
inverse AST proof. No original source or registered bytes are modified.

`runner.py` explicitly installs the private PHY namespace before the original
dispatch, owner, PHY, projection, visual, score or selection entry. The real argv
records the wrapper and retains original script/action adjacency. It verifies its
own and adapter SHA; original entry SHA values are frozen. The owner verifies the
new code bindings before routing every stage. The visual process still receives
only its original source-free registration, not the master or calibration data.

`recover_registration.py` accepts only the two known partial files and refuses
any scientific attempt or additional registration file. It reuses the original
builder with four exact source substitutions and a private write adapter. Existing
nominations/PHY files are compared byte for byte and never rewritten; missing
files are exclusive writes. The original deadline `1791954813.2221007` and 43,200
second wall budget are inherited. Original root review and the independent
metadata review are both explicitly pinned. A further failure is preserved; this
entry has no retry, scientific run or automatic follow-on.

CPU tests: `python -B -m unittest discover -s experiments/layered-mismatch-raw-20261013/r1_source_identity_revision_v1 -p 'test_*.py' -v`.

The explicit science dispatch, only after successful registration and independent
review, must use the original `r1_execution_resource_v2/dispatch_once.py` through
`runner.py --runner-sha256 SHA --adapter-sha256 SHA --original ORIGINAL_DISPATCH
--expected-registration MASTER_SHA`. Never invoke the original dispatcher/owner
directly for this registration. No new scientific budget, models, noise namespace,
resource policy, GO criterion, RX information or timing configuration is introduced.
