# In-band image transmission experiments, 28 November 2026

The six mechanisms, endpoint metrics, permitted fitting, seeds and resampling unit are specified in [protocol.json](protocol.json). The results, resource ledger and publication comparison are in [the research report](../../reports/inband_image_frontiers_20261128/README_zh.md).

The four roles run against the frozen PFS assets pinned in `study.py`, `single_link.py` and `access.py`. They read the registered source identities and the prior four-user adapter. Each role has its own output directory. The original run executed the following commands from a normal-file copy of this directory under `code/liulu/analysis/var_comm_inband_image_frontiers_20261128`, using the project’s frozen visual Python environment. The launcher records the child identity, bound file digests and actual exit code in the separate runtime directory.

```sh
python -B launch.py activity 0
python -B launch.py separation 1
python -B launch.py frontend 2
python -B launch.py drift 3
```

Once the four children have exited successfully, the numerical release can be generated with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B analyze.py
python -B audit_release.py
```

To inspect the deposited statistics, use the CSVs directly. The shared runtime paths address this study's registered inputs and outputs. Launchers use exclusive creation to protect existing results. An independent replay needs fresh work and runtime directories and access to the pinned models, source images and environment. The released event tables and statistical code also support a CPU-only reanalysis.

The subsequent receiver-side amplitude prediction control was registered after the first six experiments. Its [protocol](source_predictor_protocol.json) fixes both regressions and reuses the exact channel events and native scores. The recorded command was `python -B run_source_control.py 2` after the first-round audit. Its four-way comparisons and model coefficients are supplied in the report's `source_predictor` directory.

