# Active entry after metadata and GPU identity review

This file supersedes the draft command names and UUID visibility sentence in README.md. Historical drafts are retained unchanged and must not be executed.

- Metadata preparation: `prepare_v2.py`, already completed. It verifies only the relocated static `completion` and `table` files, with the original SHA equality preserved. The v1 missing historical-path failure is retained.
- Registration: `register_v3.py`, root review required as already specified. The CLI still takes `--gpu-uuid` as the **NVML** selection. It uses a separate CPY subprocess and CUDA Driver API to query actual CUDA ordinal, UUID and PCI without Torch, model loading or context creation. NVML and CUDA are joined strictly by PCI, and the actual mapping is saved. The namespaces are not assumed equal.
- Stage control: `nc_control_v2.py {phy,receiver,metrics,delivery} <registration-SHA>`. CUDA visibility uses the mapped CUDA ordinal with `CUDA_DEVICE_ORDER=PCI_BUS_ID`; resource checks still use NVML UUID; metric identity checks use the CUDA UUID; the original RX implementation also checks the physical PCI address.
- Necessary additional tests: the four `test_static_relocation.py` cases and six `test_gpu_mapping.py` cases. Do not rerun the already passed 23 core synthetic cases.

Neither these changes nor registration invoke scientific stages. Root must inspect the closed CPU receipts, unchanged math/source-transform proof and all current file descriptors before a unique actual launch. All historical failures and prior source bytes remain.
