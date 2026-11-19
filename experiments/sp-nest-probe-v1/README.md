# Superposition NeST-Com broadcast probe

Pilot50 reached its preregistered **NO-GO** decision. At a fixed 1,024-symbol budget, superposition improved the equal-user DINOv2 and DISTS over time division at both receiver pairs. Relative to the best single codeword, DISTS fell by 4.58% at 4/19 dB and rose by 3.00% at 4/13 dB. The probe ended before SP2. [Full report](../../results/sp_nest_probe_v1/REPORT_zh.md)

## Reproducing the measurement

Use the archived project source at WCL base `564c18059dbe6c80ada4c111df01bdd8320fc52b` together with this experiment directory. The run used the registered 50-source manifest, original NeST tokenizer, VAR prior, fitted decoder, native source arrays and integer arithmetic coder. The numerical routes and original commands are recorded in the delivery. Each stage writes to its own fresh output directory. [Model identity](../../results/sp_nest_probe_v1/a800_pilot_v1/source_identity.json) · [Run arguments and receipts](../../results/sp_nest_probe_v1/a800_pilot_v1/execution.json)

1. `run_sp0_radio.py`, `qualify_split.py` and `qualify_source_frame.py` check the demapper limit, actual LDPC/CRC/SIC and independent arithmetic decoding. `verify_pfs_cache.py` verifies the Pilot50 source and cache identities. `qualify_native_cuda.py` checks compatibility with the historical probability route. The A800 run recomputed the codec outputs with a matching independent receiver on that device.
2. `a800_clean_pilot.py` emits canonical B/E streams, actual single-prefix lengths and native FP32 reconstructions. `a800_pilot_quality.py` scores the clean, fallback and gray states. `sp1_rx_catalog.py` indexes previously verified independent decodes by actual received bits, public endpoints and recovered coarse context. The receiver has no source labels, transmitted tokens or encoder probability tables.
3. `broadcast_plan.py` fixes the source roster, noise, source-bootstrap bank, 300 single candidates, layered families and mapping to physical transmissions. `profiles.py` constructs the public SP directory. OMA carries its time split in its public profile. The receiver reads the protected header at its actual SNR.
4. `a800_broadcast_phy.py` runs the registered single, superposition and OMA families. Its CUDA LDPC encoder is checked against CPU code bits for each layout before evaluation. Output records store received statuses, token identities and measured energies. A valid-looking unfamiliar bitstream is retained for independent decoding. `continue_pilot_phy.py` was the finite supervisor for the two controls.
5. `score_broadcast_pilot.py` joins the received identities to four native-image metrics, selects on DINOv2 at each user weight, and evaluates the prespecified paired gate. `export_pilot_evidence.py` exports scientific tables, asset identities and process receipts. Raw wave-level records and reconstruction arrays reside in the PFS run directory in the resource ledger.

The delivered tables can be checked and the figure rebuilt directly from this branch with NumPy and Matplotlib.

```sh
S=experiments/sp-nest-probe-v1
D=results/sp_nest_probe_v1/a800_pilot_v1
python "$S/validate_pilot_delivery.py" "$D"
python "$S/plot_broadcast_frontiers.py" --evidence "$D" --out "$D"
```

The plot uses the observed non-dominated policies for each metric. Its circled operating points were selected on equal-weight DINOv2. The available 50 sources were used for selection and pilot assessment. See the conditional source-paired intervals and full operating-point table in the report.
