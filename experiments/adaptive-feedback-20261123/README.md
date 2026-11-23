# 有限反馈与自适应图像传输

实验由三部分组成。冻结 SwinJSCC 上比较有限反馈格式，公开手机日志用于检验反馈跟踪，原 NeST 接收结果用于评估延迟反馈下的数字帧决策。完整结果和研究定位见[中文探索报告](../../reports/adaptive_feedback_20261123/README_zh.md)。

## 图像重建

使用仓库同级 `direction-followups-20261123/common.py` 定义的 PFS 资源、冻结 Swin 主干、收发联合校准器与原生共用测试图。依赖为此前发布的 `dir2_zpac` 实现、离线 DISTS 权重、PyTorch 和 NumPy。研究根目录是 `code/liulu/analysis/var_comm_adaptive_feedback_20261123`，临时文件进入 `liberai-tmp/liulu/liulu-var-comm/adaptive_feedback_exploration_20261123_v1`。输入权重和源图清单的 SHA-256 在报告内可查。[输入身份](../../reports/adaptive_feedback_20261123/image/registration.json)

`run.py` 依序登记、拟合反馈量化和预测参数、生成配对信道、在一张空闲 GPU 上评分、按源图汇总。已有这份运行记录时可直接读取 CSV。重现时为 `ROOT` 和 `TMP` 指定新的个人普通目录，使用相同源图清单与资产路径，保留已完成记录。各阶段的命令为

```text
python -B study.py register
python -B study.py calibrate
python -B study.py prepare
python -B study.py score
python -B analyze.py
```

为核对反馈因果性，信道准备器只把截至已送达时刻的观测交给反馈编码器。发送端预测器仅接收报告整数序列、公共时钟和冻结系数。脚本还改变未到达的样本，并验证已送达符号和当前预测逐值一致。每个四时隙周期内，四种等预算格式分别实际构造四个 1-bit、四个 1-bit 升降、两个 2-bit、一个 4-bit 离散符号。各自在反向传输层的包数和空口映射需要单独建模。

## 手机日志

取作者数据仓库固定提交中的 `5G-production-dataset.zip`，按协议校验 SHA-256，放入本研究的临时目录。在完成图像无关的量化器冻结后，分别运行 `python -B trace_replay/replay.py` 和 `python -B trace_replay/replay_clock.py`。前者处理严格连续采样，后者按真实秒钟处理短缺口。初次严格回放保存了全部逐会话结果后，静止组为空使汇总脚本退出；实际交付使用 `python -B trace_replay/field_summary.py field` 汇总原始输出。修订版入口可以处理空组。原始手机数据留在临时存储，交付物只含会话汇总、覆盖情况和数据身份。[数据仓库和许可](https://github.com/uccmisl/5Gdataset/tree/306d92eba81cb4f1195d8b8eba625666a02a10ec)

## 原 NeST 数字帧

`python -B digital_replay/replay.py` 从原 N1024 校准登记中读取逻辑帧、接收状态、图像对和 DINO 分数，并与已发布的 1,200 个 NeST 候选得分逐项核对。只对确定的 50 张选表图求策略，另外 50 张用于本次回放。`python -B digital_replay/risk.py` 接续此表和源划分，在冻结平均风险上限后分别求解四个小型线性规划，再交叉评估信道持续性失配。后者使用 SciPy HiGHS。实际执行时为矩阵读取函数增加了可选的审计输出目录，以分开两次回放的运行记录。输入矩阵身份逐项比对一致。数据输入路径、哈希与源图顺序均已保存。[输入矩阵](../../reports/adaptive_feedback_20261123/digital/registration.json) · [源图划分](../../reports/adaptive_feedback_20261123/digital/split.json)

从仓库根目录执行 `python -B experiments/adaptive-feedback-20261123/figures/plot_digital.py reports/adaptive_feedback_20261123` 可重画报告图。绘图只读取已冻结的配对差值和拒收结果。
