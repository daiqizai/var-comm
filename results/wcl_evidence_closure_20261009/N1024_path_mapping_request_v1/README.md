# Unsealed local report/figure path map request

All 314 files from N1024_delivery_index_v1 are requested for actual remote SHA verification, plus its nine own files and six explicitly bound report/figure scripts: 329 files total. 277 entries are eligible for the local report/figure output path map; 52 are verification-only. The copied frozen policy, remote-authored scientific exports and historical scientific receipts are never path-resolved through this map. No scientific gate, checkpoint, model, input image, registered source record or .research input receives a mapping. Source references inside each artifact are preserved byte for byte.

The examples v2 local script has remote name plot_t5_completed_examples_delivery_279a229c.py. The remote historical same-name script is not replaced or silently mapped. Other remote paths are the exact copied delivery paths explicitly listed by the source index, not guesses about old scientific inputs.

Status is UNSEALED. No successful remote verification or copied-file availability is claimed before the remote process returns. A missing or mismatched file yields a failed receipt, and seal refuses it. Existing scientific/report outputs are never edited; the verifier reads only the enumerated files and writes only a new receipt.

Upload the request and n1024_report_path_map.py through the root-owned connection. Root runs:

```text
python experiments/wcl-evidence-closure-20261009/scripts/n1024_report_path_map.py verify --request <uploaded_request.json> --request-sha 6aef816d80d5d9a48b9339d09b839464ebc96bd6f256aef87ab68d0e804bf1f9 --receipt <new_remote_receipt.json>
```

Download that actual receipt, independently obtain its remote SHA, and run locally into a new directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/n1024_report_path_map.py seal --request <this_request.json> --request-sha 6aef816d80d5d9a48b9339d09b839464ebc96bd6f256aef87ab68d0e804bf1f9 --receipt <downloaded_receipt.json> --receipt-sha <actual_remote_receipt_SHA> --out <new_sealed_mapping_directory>
```

This map is exclusively local-authored N1024 report/figure navigation and restore provenance. It is not an authority to redirect any scientific dependency, relax any hash check, reinterpret a science gate, or declare T6 complete. No raw C:\ reference embedded in a report is edited or replaced. No SSH, model, channel, bootstrap or timing calls are made by this workflow.
