# Explanatory fidelity example, not a success-rate estimate

The top row (index0) in [the unchanged first-two figure](figures_v2/fixed16_first_two.png)
is also a useful explanatory failure example. The ordinary figure remains the
first two sources in the original registration; it was not chosen to favor a method.
For this additional annotation, the explicit descriptive rule is minimum HiFi PSNR
over all16 (index0,16.9861dB). This is post-score explanatory selection, not a new test.

The source contains a person holding a fish. The HiFi reconstruction has altered
facial and shirt details; its texture reconstruction does not ensure fidelity.
Raw also changes the face/expression and shirt lettering. The visual example does
not establish identity preservation by raw or an overall ordering. All four native
outputs, all16 sources and all numerical scores remain included. Only Swin/HiFi
share an actual received observation; raw uses its preserved6201 noise.
The figure uses uniform display quantization, with no local enhancement. No new
image recovery, metric or population hallucination-rate estimate was made.
