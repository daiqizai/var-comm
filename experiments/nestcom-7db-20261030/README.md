# Bounded7dB calibration and fixed development comparison

User-authorized scope: existing100 calibration sources, five declaredMCS, at most2500 new calibration digital frames and12 GPU-device-hours; freeze before comparing the same Pastries95/Dog21 using WHOLE,NeST,Ada,ARPC,LatentJSCC,SwinJSCC andBPG. No training or common500 expansion.

The pre-call protocol is in `results/nestcom_7db_20261030_v1/protocol.json`. Its exact100source identity joins are in `calibration_sources.json`; the two development sources are disjoint. New calibration is2000frames forNEST/WHOLE/ADA/ARPC. BPG reuses its verified original7dB profile3006 from100sources×3noise and its originalPSNR objective; the other four use the predeclaredDINO objective with one calibration noise draw per source. These objective/population differences must remain explicit.

`phy.py` creates actual N1024 frames from independently terminated cached streams and exports public received inputs. `rx_native.py` invokes the closed received-only NeST/Ada route; `rx_arpc.py` invokes the closedARPC route. `control.py` records one actualwait, enforces finite worker time and a cumulative maximum reservation of43200GPU seconds, and never automatically retries. Any original failure remains immutable.

The initial ARPC entry used a nonexistent venv executable. Its failedPopen had no child and no scientific call; `arpc_entry_v2.py` uses the already qualified systemPython plus frozenARPC package path. The exact before/after entry change and originalfailure are recorded separately. No mathematical changes or successful-call repetitions were made.

`dino_entry.py` uses the originalDINO evaluator with a hash-checked checkpoint storage resolver. `freeze.py` reports every candidate before freezing oneMCS per method. `dev_dispatch.py` permits exclusive first starts of the fixed development source,PHY,RX andmetric phases only after freezing. TX andRX remain separate processes. `prepare_development.py` checks exact source/cacheidentity; ARPC container repacking preserves its independently encoded scale blocks. `dev_aux.py` preserves the qualifiedBPG codec resource shim and continuousJSCC numerical paths.

`prepare_scores.py` joins nativearrays to sourcepixels. `dev_scores.py` uses the originalFP32LPIPS-Alex and ensembleDreamSim loaders. `export.py` checks real exits/ledgers and exports lightCSV/JSON/PNG; arrays remain onPFS. `draw.py` produces PDF/PNG/editableSVG galleries at full field of view and an honest4/7dB fixed-source comparison. No bootstrap or population claim is introduced.

Scripts are one-shot by design; do not rerun a completed registration or hot-edit code bound to an active stage. Inspect actualwait/PID/startticks and ledger before advancing. Historical10dB artwork remains unchanged.
