# Joint/R-only控制激活决定

原六臂RX-only已经完整5000更新、全校准、全development回放及CPU分析；reference资格核查PASS。原RX评测receipt SHA：`b2e5711f350937fe98ece94fdd4bf671ff0745e98b5aad04002816c7ba57ffe1`，分析receipt SHA：`be74cc97b7bb445cb4ef95027af3008b34e214381451a4083973f173939e30e3`。

当前多尺度创新量主区间未胜single或full-grid；保留其机制证据而停止追加该候选。全网格迭代有局部图像改善，但低SNR主区间仍落后强系统，高SNR也未超同WeTok数字链，不将它改称next-scale成功。

据已登记`docs/protocol.md`，现在开始三基础结构Joint E/R控制，首1000更新里程碑、原计划5000。不是从RX-only已训练5000的末端开始，而是从**与其相同的原7000 single父点**复制同一初值，使用同fresh Adam规则和相同global data7000之后的数据/增强/SNR/噪声序列。这样Joint5000最终对R-only5000，不因多训练一步占便宜。

只改变通信E是否被优化；视觉模型、204×30几何、loss/学习率、micro1/effective4、无反馈/无真接收前缀和N3060/E6120全部不变。E/R图像梯度已在真实GPU验证，启动不以未训练质量胜出为门槛。

原R-only六臂全部保留为未来强参照并重测时延；Joint若有条件结构增量，还需Joint普通全网格对照。新实验不访问新holdout，不把DINO加入训练或选模。

已有profile三Joint臂主体前/反向约0.321 GPU小时/1000更新；首1000还包含起点和终点完整校准、一次监控及装载/保存，预计约0.65–0.8小时。实际速度和温控以运行记录为准，不能当总工期保证。

当前原RX GPU流程已自然结束且无其它计算任务，允许本地启动；不停止其它用户进程，不改驱动、MPS、时钟或功率设置，不提交/push。
