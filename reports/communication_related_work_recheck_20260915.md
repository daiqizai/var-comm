# 本轮相对既有工作的定位

2026-09-15重新核对ARPC的ICLR 2026正式页面和Ada-TokenCom的arXiv v1全文；与先前保存的论文/代码核查一致。不是作者系统的同协议完整复现。

- **ARPC**：已有next-scale概率、渐进前缀、后缀生成和算术熵编码。我们的整帧对照采用同一官方VAR概率，不能据此声称首创生成式源压缩；其视觉表示和压缩任务与本项目不同。
- **Ada-TokenCom**：已有AR概率算术编码、源率/MCS联合适配及质量—资源优化。其CRC和重传资源有计费，图像质量统计以成功交付为条件；语义条件的少量bit在其报告中忽略。不能将它说成只优化BLER、只发固定索引或不计失败资源。
- **本轮具体问题边界**：一次固定N/E、无重传、真实header、保留CRC失败候选、所有失败输出计入图像质量。比较同表示/同实际FEC下，不同源编码与预定模式选择目标的后果。

这不是“前缀生成”“next-scale”“源率/FEC适配”或一般期望失真最小化的新颖性声明。若普通熵编码＋可靠性策略与质量策略采取同样动作，则没有独立的新调度机制；更多实验或新网络不能改变这一逻辑。

来源：ARPC正式页面 `https://proceedings.iclr.cc/paper_files/paper/2026/hash/a255ea760cfedf42a988d9de6ec902cf-Abstract-Conference.html`；Ada-TokenCom `https://arxiv.org/html/2608.28086v1`（2026-08-28 v1）。此前完整差异与读取范围见`reports/convergence_report_20260915.md`第八节。
