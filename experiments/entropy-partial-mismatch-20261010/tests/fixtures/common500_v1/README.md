# Pinned common500 metadata fixtures

These three files preserve the original bytes used by `scripts/test_mismatch_plan.py`.
They make the metadata tests independent of private `.research` restoration paths.
All original SHA256 checks and the 900 channel-counter comparisons remain enabled.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `plan.json` | 31684 | `1ab63b0e0bdde865d677467232c956ebbae746239ac300dd7723e22dbe8c7bce` |
| `manifest.json` | 590608 | `b27128fb8eedee7f2cc74bed25c63e39e8add8c938c76d449f85c4b0d5a51e09` |
| `raw64_unified500_plan.py` | 7219 | `1acc294032f46166ff2b714b6d47de1ac7aa8bb37619c8b75729f801ac249fec` |

The JSON files contain the existing common500 ImageNet source identifiers, frozen
configuration metadata, and hash/path descriptors. They contain no source pixels,
latent arrays, model weights, credentials, or new scientific measurements. Historical
absolute paths are provenance strings; the tests do not open their referenced assets.

The original Python fixture defines frozen scheduling and counter functions. The
test checks its exact bytes before evaluating those definitions and invokes only
`frame_counter` and `canonical`. Its `standard_noise` function is never called;
NumPy is imported only inside that unused function. No model, PHY simulation,
image inference, metric computation, bootstrap, or policy selection is performed.

Original provenance: `main_raw64_20261007/take_over_v1/current/`
`unified500_raw_execution_v1/plan.json` and
`common500_source_assets_v1/manifest.json`, plus
`main_raw64_20261007/take_over_v1/unified500_raw_v1/raw64_unified500_plan.py`.
