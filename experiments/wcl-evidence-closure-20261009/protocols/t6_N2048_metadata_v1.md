# N2048 resource planning and source exclusion audit

This is an independently named metadata stage. It does not authorize or launch channel decoding, reconstruction, neural training, quality selection, or a test run. N1024 conclusions must precede the T6 scientific stage and the entropy-family choice remains unset.

The fixed budget is 2,048 complex symbols: 68 paid header symbols and 1,980 body/padding symbols. The raw branches use raster raw12 tokens, one body group, CRC16, actual original Sionna LDPC, QPSK/16QAM/64QAM and KEEP. Known QPSK frame padding consumes symbols and energy; it is neither silence nor additional parity. There is no frame-specific QAM normalization. The mean constellation energy is two, and this metadata stage reports no fabricated actual waveform energy.

The planner imports the original `main_action_space.py` (SHA `7ca5809c00f8fd5ac0040d0abd664e1673cee71287bbe383e07fd1039a87e28d`) and original `profiles.Planner`. Only the private imported module's N/body constants and modulation dictionary are changed for this new budget. Original files are not changed. The five nominal rates are 1/3, 1/2, 2/3, 3/4 and 5/6. Both nominal-length and full-budget protection layouts are considered. The original finite rule is retained for every scale, MCS and allocation mode:

`K ∈ {0, floor(L/4), floor(L/2), floor(3L/4), maximum actually legal K}`.

The maximum is found by the original descending actual-backend feasibility check, without assuming that LDPC lifting feasibility is monotone. A complete next scale is canonicalized to the corresponding whole prefix. Whole scales 1 through 10 are admitted when feasible, and every whole action belongs to the PARTIAL family as well. The complete ten-scale payload is 680 tokens, 8,160 source bits and 8,176 information bits including CRC. These lengths alone do not establish actual code feasibility.

`t6_plan.py queries` emits a 7,372-tuple integer resource-query superset. This is explicitly **not** the scientific candidate set or a list of legal layouts. `plan` calls only the existing actual encoder constructors, blocks encoder forward/decoder operations, records every admitted or explicitly unsupported resource query, and uses the unchanged finite generator to produce the final action table. Rejected constructor layouts stay in the evidence; unexpected software errors stop the stage. The new public catalogue is deduplicated by exact wire parameters, assigns at most 4,096 new paid profile IDs, and preserves nominal-rate/allocation aliases. It is a separate namespace and does not make old N1024 or historical N2048 receptions automatically reusable.

Run metadata preparation on the actual repository after uploading the new script:

```text
<original CPU Python> -B experiments/wcl-evidence-closure-20261009/scripts/t6_plan.py prepare --root /home/liulu/projects/VAR_COMM --environment-request <registered T2 request.json> --out /home/liulu/projects/VAR_COMM/outputs/WCL-EVIDENCE-CLOSURE-20261009/T6_N2048_metadata_v1 --deadline-unix <explicit future Unix deadline>
```

Then invoke `t6_plan.py plan --request <new output>/request.json` using the registered original LDPC Python with `CUDA_VISIBLE_DEVICES` empty. The metadata process uses two CPU threads, nice 15 and at most 7,200 seconds. The parent schedules it; the script launches no background task. Its packet and GPU budgets are both zero. Interrupted metadata planning resumes exact bound constructor records. Completed metadata receipts are verified and returned without repeating constructors.

Actual output files are `candidate_catalogue.json`, `finite_enumeration.json`, `legal_actions_metadata.csv`, per-query `plans/*.json`, and `completion.json`. The completion status is `T6_RESOURCE_METADATA_COMPLETE_NOT_PHY_QUALIFIED`; a constructor-admitted row is not a successful packet roundtrip or a performance result. A later independent protocol must qualify the changed N2048 physical implementation and register calibration and evaluation budgets before scientific execution.

The formal test design is fixed at 100 sources × 4/10/19 dB × three noises × three digital methods = 2,700 method-frame conditions. Calibration and source preparation are additional, currently unregistered work. Both raw granularities will use the same frozen generation rule and original calibration population. The third family's selection must come from T1's N1024 calibration conclusions, never N2048 test outcomes. No entropy family, noise seeds or confirmation source IDs are selected by this metadata script.

## Existing source evidence

`t6_source_audit.py` reads explicit manifests for original train20k, calibration1000, development100, holdout500 and named historical populations. It canonicalizes ImageNet validation image-number aliases so a class/path/extension change cannot make a used image look unused. It reads no pixels and never selects a source subset.

The local audit ran on the actual available manifests. Their conservative exclusion union contains 23,500 unique source identities. The only original-image pool explicitly represented in the inspected split manifest contains 3,000 previously registered ImageNet validation sources; all are excluded by the known-history rule. Thus the current audit admits zero unseen sources and writes `BLOCKED_NO_UNSEEN_POOL_SOURCES`. It does not prove a global source-use inventory, physical image availability, or absence of content duplicates under different identities. A full unused source pool is currently undocumented here, so no independent confirmation100 list has been created. Existing seen data may only be used later under an explicitly registered same-source-extension label, never silently renamed a new blind test.

Local audit outputs are `.research/wcl_evidence_closure_20261009/t6_metadata/source_audit_v1/source_audit.json` and `pool_exclusions.csv`. Query preparation outputs are in the neighboring `query_superset_v1` directory. These are actual metadata products; the actual Linux constructor stage and all T6 science remain pending.
