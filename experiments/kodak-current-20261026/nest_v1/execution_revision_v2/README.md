# Pre-launch downstream entry-name adaptation

The source-only v1 revision completed all 24 TX traversals, 600 terminated streams, and no tokenizer or decoder calls. Before any downstream launch, inspection identified that original source has main(), while original PHY, receiver and delivery use run(). No downstream stage was attempted with the incompatible dispatch.

This additive v2 invokes original run() only for PHY, receiver and delivery. Source execution is prohibited; its already closed v1 waits are verified with the exact v1 revision SHA. The source bytes, outputs and successful calls are never changed or repeated. Other behavior is byte-equivalent to the v1 metadata control code except its own revision path, explicit prior-revision binding and role-name dispatch. The physical GPU numeric-zero/PCI mapping is unchanged. All original scientific code remains pinned and unmodified.

No additional scientific budget or automatic retry. Final delivery retains the original parent registration, with v1/v2 execution revision and failed initialization lineage recorded separately.
