# 固定m8配对续训的远端准备

状态：**本机只准备；训练、视觉模型推理与GPU测速均未开始。**

论文主线：有限带宽与总能量约束下，利用next-scale条件结构设计学习式图像传输与恢复。
这轮先把现有m8映射训练清楚，不换骨干、不扩模型、不加全尺度/功率/FEC/反馈，不重训parallel。

## 准备入口

- 代码仓库：`../var-next-scale-comm/`。
- 用户操作指南：`docs/REMOTE_M8_HANDOFF.md`。
- 独立协议：`reports/prefix_refinement_remote_protocol_2026-09-11.md`。
- 配置：`configs/prefix_refinement_remote.yaml`。
- 计划/显式执行入口：`scripts/remote_refinement.py`；不加`--execute`不运行模型。

A保留MSE+0.01LPIPS+0.001CE；B先CE+0.1state，后MSE+0.01LPIPS+0.01CE+0.01state。
同冻结epoch2、同Adam状态与步数10000，LR3e-5、有效batch4，原自身历史、hard/ST与68+2992账本不变。
初始每分支新增10000更新；预算确认前可在远端根据无更新profile另建成对预算配置，不中途给两分支不同预算。

每500步固定100图监控；0/2000/4000/6000/8000/10000步完整1000图校准，LPIPS选模附各SNR PSNR退化≤0.2 dB，监控子集不选模。
评测保留旧起点、数字m8、自适应和感知Deep。Deep0+3060与其余68+2992分别校验；5/6补充单列，旧服务器时延不混入新服务器排名。

资产转移清单587项、8.025 GiB：只保存清单与校验，没有在本机复制大型权重/数据，也未连接远端。
原模型、state/loss核心实现和旧工作区训练源码未修改；新仓库的控制脚本/辅助工具变更有SOURCE_MANIFEST记录。
CPU合成检查不等于GPU训练验证；真实环境、完整起点复现和费用须在目标服务器检查。后续功率/全尺度是独立实验，不要求本轮必须全面打赢数字才允许研究。
