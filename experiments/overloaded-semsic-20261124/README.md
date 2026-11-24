# Four-user interference replica study

The first experiment compares local receiver interventions with an unchanged transmitter, observation and frozen model. The follow-up additionally fixes the RGB receiver's gates. The two registrations are `protocol.json` and `gate_protocol.json`. The report includes negative results, event-level scores and the editable short paper. [Results and archive](../../reports/overloaded_semsic_20261124/REPORT_zh.md)

## Inputs and evaluation

Use the PFS runtime for the released experiments with PyTorch 2.11.0+cu128 and the same SwinJSCC, DISTS, source arrays, and seed-specific weights. The exact hashes are in the admissions and the resource ledger. The canonical study directory on PFS is `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/analysis/var_comm_overloaded_semsic_20261124`. The evaluation imports the original code from revision `faaa2799f67593f1fe7846a4b421701a19d81f4d`. Supply a study clone with that Git object to the preparation script. It exports ordinary files and checks each insertion against the original text. The Git report stores small score tables. Checkpoints and image tensors remain on PFS.

```sh
python -B prepare_snapshot.py "$EXPERIMENT_REPO" "$SNAPSHOT"
CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B evaluate_replicas.py "$SNAPSHOT" "$WORK/events" "$RUNTIME" --gpu "$GPU"
CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B rgb_fixed_gates.py "$SNAPSHOT" "$WORK/events" "$WORK/gate_events" "$GATE_RUNTIME" --gpu "$GPU"
```

The evaluator creates each score directory once. The small `wait.json` files passed to the statistics scripts are the corresponding process owners' actual exit receipts. Both successful receipts are included under `execution/` in the results. Use the original image identity, group ordering, conditional seed draws and frozen receiver arm names when reconstructing paired estimates.

```sh
python -B analyze_replicas.py protocol.json "$WORK/events" "$SNAPSHOT" "$WORK/execution/primary_wait.json" "$WORK/analysis_new"
python -B analyze_fixed_gates.py "$WORK/events" "$WORK/gate_events" "$WORK/analysis_new" "$WORK/execution/gate_wait.json" "$WORK/analysis_new/gate_paired.csv"
python -B plot_evidence.py "$WORK/analysis_new/paired.csv" "$PAPER"
python -B make_paper_tables.py "$WORK/analysis_new" "$PAPER"
typst compile "$PAPER/manuscript.typ" "$PAPER/manuscript.pdf"
```

The source tables and vector figure accompany the manuscript. The paper was compiled with Typst 0.15.2 and rendered page by page. `archive_portfolio.py EXPERIMENT_REPO DEST` reconstructs the portfolio index from pinned research commits and the original PFS paths. Use an empty destination when rebuilding the index.
