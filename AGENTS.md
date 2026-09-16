# Public research snapshot

- Read `README.md`, `docs/RESEARCH_STATUS.md` and `docs/REPRODUCIBILITY.md` first.
- Keep the focus on image communication under explicit channel-use, energy and receiver-information constraints.
- The fixed-m7 hybrid study is closed. Historical launch instructions do not authorize new training, architecture search or new holdout access.
- Preserve negative results and distinguish mechanism evidence from system gains.
- Do not change published CSV values, frozen model choices or selection probabilities to improve a conclusion.
- Raw source/image data, model/optimizer checkpoints, secrets and third-party vendor trees are not publication content.
- The user-authorized external update includes preregistered reconstruction/comparison figures in `results/external_baselines/figures/`; panels can contain source references. This does not authorize standalone source files, datasets or raw pixel arrays.
- `tools/reproduce_results.py` is CPU-only statistical reproduction. Do not describe it as image-metric recomputation or training reproduction.
- Reports/configurations contain sanitized historical paths. Original and published hashes are separately recorded.
- Do not force-push, bypass hooks or overwrite remote work. Create commits only within explicitly authorized tasks.
- Standing user instruction: immediately push every authorized commit to its configured remote, verify remote inclusion and report any failure. Do not ask for separate push confirmation or mistake a local commit for a completed upload. This does not authorize automatic commits of unfinished/raw assets.
