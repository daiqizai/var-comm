# Action-space upper-bound audit

[Full report](../../reports/action_space_upper_bounds_20261005.md). CPU-only audit; no new scientific evaluation.

M1_AUDIT.json records all 30 current M1 N/PHY/SNR cells and their exact saved-grid coverage. M1_MISSING_ACTIONS.csv lists 24 distinct actions absent from the union of current registered grids. The low-SNR m9 scope difference is recorded separately. N3060/N4084 rows in M1_RESOURCE_ACTIONS.csv are hypothetical resource bounds, not measured M1 results.

The independent audit script is preserved as text to avoid modifying frozen experiment source inventories. Run it on the original server checkout, which retains the referenced experiment outputs:

```sh
python results/action_space_audit_20261005/audit_m1_actions.py.txt --root . --output outputs/ACTION-SPACE-AUDIT-20261005/recheck --strict
```

The current strict check returns code 2 for the known gaps; this is an intentional finding, not a runtime failure. Without --strict it writes the evidence and exits 0. It neither loads models nor launches experiments. Input identities and the source-registration matches are preserved in the JSON evidence.
