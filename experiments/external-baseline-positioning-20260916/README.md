# 作者外部基线源码快照

只评测公开预训练SwinJSCC、ADJSCC及其HiFi-DiffCom恢复，不训练新模型。

- 设置/权重SHA：`configs/protocol.json`；数字预算：`configs/digital_common_budget.json`。
- 结果：仓库根`results/external_baselines/`。
- CPU复算：仓库根`tools/reproduce_external_results.py`，不会启动GPU队列。

不包含作者vendor、权重、venv、源数据或全量重建数组。重演须取得固定commit的代码、公开权重、合法数据及历史组件，并登记新环境资格。

作者环境是独立torch1.12.1+cu116、torchvision0.13.1+cu116、numpy1.23.5、Pillow9.5.0、timm0.4.12；指标/数字参考用原torch2.11环境。不承诺clone后一键训练。

`prioritize_non_diffusion.py`是原工作区单次排程源码，不是通用部署命令。不要按历史PID操作新机器进程。发布时HiFi尚未完成。
