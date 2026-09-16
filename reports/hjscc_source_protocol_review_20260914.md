# HJSCC强对照：源码与协议资格准备

2026-09-14只读核对原论文、作者仓库及8个小型源码文件，共43693字节。没有下载权重/数据，没有执行作者代码、安装依赖或启动新GPU实验。当前Joint训练和注册评测保持不变。

## 版本与本轮结论

arXiv落地页当前仍为2408.16340v5，最后修订2025-02-27，标注AAAI2025接收。作者仓库本次`main`定位到`bea5c6c7ec42ea62e5afac307d37f9e20129cc6d`，该提交日期2026-03-02；不是把今天的读取时间当成论文或代码发布日期。

**HJSCC仍是应保留的强系统对照，但当前公开示例不能不加适配就放进本项目每图固定3060-use排名。** 以下是适配和资格条件，不是对作者方法性能的否定，也不是本项目已经胜过HJSCC。

源码快照与Git blob SHA、文件SHA256见`outputs/HJSCC-SOURCE-REVIEW-20260914/manifest.json`。递归目录读取和raw域连接各有一次失败，记录保留；最终从官方GitHub blob API取得源码，并核对Git对象哈希。没有使用不完整下载。

## 已定位的接入要求

下列代码路径均相对于`outputs/HJSCC-SOURCE-REVIEW-20260914/author_source/`。

| 核对项 | 实际证据 | 接入本项目时必须处理 |
|---|---|---|
| 变长及分组布局 | `HJSCC/layer/HJSCC_EncDec.py:58`由Encoder产生mask/index，再将index直接给Decoder | 独立收发边界必须说明这些布局信息如何到达RX；计费并保留失败样本，不能依赖Python对象免费传递 |
| CBR口径 | `HJSCC/layer/HJSCC_EncDec.py:63`统计实际复符号，除以图像`C×H×W`；`HJSCC/net.py:324`相加 | 对256×256 RGB，3060 uses对应CBR **0.01556396484375**。不能把CBR与raw bits、实坐标数或`H×W`口径混用 |
| 逐图固定资源 | 当前长度由源相关mask决定，`channel_usage`是选中实坐标配对后的数量 | 先得到每图、每层真实N及控制N；平均CBR相同不证明每图满足3060。不能任意截断作者模型再当强基线 |
| 平均能量与噪声单位 | `HJSCC/channel/channel.py:25`默认复能量1，实噪声方差为`1/(2γ)` | 与本项目复能量2、实噪声`1/γ`可作一致的全局单位换算；不要只改功率或只改噪声制造3 dB差异 |
| 发送端幅度因子 | `HJSCC/channel/channel.py:28`从发送特征算`avg_pwr`，`:48`在返回RX特征时乘回其平方根因子 | 这段同进程示例含TX相关归一化尺度；尚需明确独立RX如何获得它。不能无声照搬成免费连续源边信息，亦不能据此推断所有作者实验存在同一实现问题 |
| SNR与训练支持 | `HJSCC/net.py:266`训练/测试两分支均写定10 dB；配置还单列信道参数 | 必须沿真实调用链传入实际SNR，先做原支持点回放；不能给10 dB示例零样本改标签就当充分训练的1/4/7/13/19 dB强对照 |
| 图像指标 | `HJSCC/net.py:331`返回loss/PSNR/CBR等标量，图像留在内部 | 适配应导出实际RGB，用本项目同图/同噪声指标器计算LPIPS、DINO等；不能用作者标量与本项目不同样本均值混排 |
| 环境与权重 | `HJSCC/requirements.txt:1`固定旧依赖；root README说明两档预训练模型和对应λ | 单独环境/源码适配，不覆盖现有训练环境；先冻结取得的实际权重和训练口径。本轮没有下载、加载或验证权重 |

作者论文的“Masking and Length Information Reduction”明确讨论额外长度信息，并假设它无误传达，以容量作开销换算；不能把示例CBR直接解释成已实现有限码长header。反馈版本允许发送端使用此前接收信号，必须与无反馈版本分开。本轮首先保留无反馈系统作为待接入对照。[1]

这里发现的是**本项目严格协议下的接入工作**，不是“因为难适配就删掉强对照”的理由。HJSCC是独立视觉骨干的系统级比较，也不能替代当前同WeTok的因果对照。

## 后续执行顺序

1. 先完成正在运行的Joint5000及全部已注册强对照，检查学习映射和条件结构的实际增量；不以准备外部基线打断当前训练。
2. 真正接入HJSCC时先隔离环境、冻结原权重，复现原10 dB/原资源口径的RGB结果，不把改了接口的版本冒充作者原回放。
3. 再明确布局、幅度因子、N/E和SNR支持的收发协议。若引入固定预算适配或重新训练，单独登记为adapted baseline，给足匹配训练/校准机会；不得裁剪成弱对照。
4. 若改用平均预算比较，应另立所有方法共同遵守的实验，报告资源分布与峰值，不能改写原固定3060结果。

本次检查的root/HJSCC/layer/channel目录未发现LICENSE文件，尚未审计全部依赖许可。源码快照只作本地阅读证据，不据“公开可读”直接批准对外代码再发布。

## 原始来源

[1] 论文原文与版本：`https://arxiv.org/abs/2408.16340`；`https://arxiv.org/html/2408.16340v5`。本轮核查不是完整近期文献综述或新颖性认证。

[2] 作者仓库：`https://github.com/zhang-guangyi/HJSCC`；本地`commit.json`、`directory_inventory.json`和`manifest.json`保留具体版本及获取方式。各源文件的官方Git对象API地址与哈希在manifest中，未改作者原文件。

[3] 原始通道实现：`https://raw.githubusercontent.com/zhang-guangyi/HJSCC/bea5c6c7ec42ea62e5afac307d37f9e20129cc6d/HJSCC/channel/channel.py`。

[4] 原始传输封装：`https://raw.githubusercontent.com/zhang-guangyi/HJSCC/bea5c6c7ec42ea62e5afac307d37f9e20129cc6d/HJSCC/layer/HJSCC_EncDec.py`。

[5] 原始系统入口：`https://raw.githubusercontent.com/zhang-guangyi/HJSCC/bea5c6c7ec42ea62e5afac307d37f9e20129cc6d/HJSCC/net.py`。
