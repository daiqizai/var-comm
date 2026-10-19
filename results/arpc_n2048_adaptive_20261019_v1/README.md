# N2048 low-SNR adaptation: B0/B1 delivered, B2/B3 pending

The package-required uncertainty stop applies. This is a completed calibration
and qualification delivery, not a completed adaptive wireless comparison.

Cal100 now contains3200 clean source endpoints (2500 NeST-Com and700 ARPC).
All actual terminated source bytes were present; no TX or tokenizer rerun.
1407 existing image/score states were verified;1793 missing NeST states and
1793 DINO values were completed once.24 qualification objects used48 subframes
and96 decodes with zero repeated science. Original RX final exit1 and its
metadata-only closure are preserved in execution_receipts and outer_receipts.

At10dB five NeST-Com MCS policies remain plausible under the coarse historical
single-packet proxy: QPSK2/3,3/4,5/6 and16-QAM1/2,2/3. A maximum of two does
not cover them. B2 and B3 have not started. No new wireless quality, paired
bootstrap, seven-metric test result or final winner is claimed.

- B0_B1_delivery/REPORT_zh.md: full Chinese report, pending policies and costs.
- B0_B1_delivery/additional_costs.csv: three extra-policy ceilings relative to
  the eight-proposal plan (9000 logical objects/18000subframes/36000decodes).
- A tighter coverage plan retains only ARPC's one plausible policy at each
  SNR: nine total policy/SNR cells,27000logical/54000subframes/108000decodes.
  This still exceeds the original global limits by3000/6000/12000 and requires
  five NeST-Com candidates at10dB instead of the authorized two. NOT EXECUTED.
- calibration_diagnostics/:3200 source-level clean scores,32 endpoint means,
  all384 policy utilization rows and actual fallback endpoint distributions.
- cal100_prescreen_v1/: all384 strategies at4/10dB with frozen uncertainty rule.
- figures_calibration_v2/: actual opened PDF/editableSVG/600dpiPNG diagnostic,
  grayscale proof, English caption and root visual review. v1 is preserved;
  only the grayscale legend color was corrected in v2.
- layouts.csv and arpc_source_cost_breakdown_detailed.csv: both fragments pay
  their68-symbol header and13-bit length+16-bitCRC+96-bit reassembly record.
- qualification/: actual independent source-RX outputs and opened-image review.
- closed_receipts_manifest.json: exact copied evidence and gone-process audit.

The historical N1024 and fixed-full N2048 results are unchanged.19dB may only
be shown as its separate completed fixed-full reference. No new training,
power extension, UEP, feedback, KEEP, extra methods or heavy backups. Further
science awaits a scope/budget decision; this task does not silently drop any
plausible policy. Publication receipts are delivered separately after push.
