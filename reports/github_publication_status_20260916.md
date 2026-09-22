# GitHub发布整理状态

日期：2026-09-16。目标仓库：`https://github.com/daiqizai/var-comm.git`。

**状态：已成功推送到GitHub，远端main与本地提交一致。** 初次认证失败已解决：既有密钥使用非默认文件名，之前没有被默认SSH连接加载，不是用户没有配置过认证。

## 已完成的远端验证

- 使用已有`id_ed25519_github`认证，GitHub确认账户为`daiqizai`；没有新建密钥，也没有读取或输出私钥内容。
- 只在本发布仓库的本地Git配置中指定该密钥并使用SSH远端，没有修改全局SSH配置；密钥及本地Git配置不在提交内容中。
- 普通push成功：`main -> main`，没有force push或覆盖远端历史。
- 远端`main`及默认`HEAD`均为`16dd8d78989537d80218660c21b032407798e7ad`，与本地相同；工作树干净。

## 本地发布副本

- 工作树：`VAR_COMM/publish/var-comm/`。
- 分支：`main`。
- 提交：`16dd8d78989537d80218660c21b032407798e7ad`。
- 500个已审计文件，约21.8 MB；另有完整Git历史bundle `VAR_COMM/publish/var-comm-20260916.bundle`，约6.1 MiB。
- 未移动/删除原项目、checkpoint、数据集或失败记录。发布副本中的机器路径已脱敏，原始与发布SHA分别登记。

整理内容包括通信/PHY/算术编码源码，实验配置，历史学习参考源码，协议与正负结果报告，关键CSV、统计图、复现/依赖说明，以及不需要GPU/数据像素的结果复算工具。

没有上传模型/optimizer权重、图像数据集、缓存数组、虚拟环境、第三方vendor、论文PDF或凭据。这个仓库是代码及结果快照，不是全量训练资产包；完整GPU复现的缺失资产与迁移边界已在公开README和复现说明披露。

## 验证

- 487个导出源文件的SHA清单、500个发布文件的敏感内容/体积检查通过。
- Git暂存区blob逐一匹配发布SHA，使用`.gitattributes`避免跨平台换行转换破坏审计。
- 314个Python文件语法检查通过。
- 5项公开结果测试、8项原数字模式策略测试通过。
- 从公开逐帧表重算27000条实际/参考/期望行、1856配对区间，均值、区间、期望误差均0。
- 用Git从提交生成一个干净克隆，再次执行清单检查及完整CPU统计复算，全部通过。不依赖原图、外部权重或本机绝对路径。

## 初次认证检查遗漏（已解决，保留过程）

初次空仓库可通过HTTPS读取，但没有可用HTTPS凭据；当时推送返回：

```text
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

当时默认SSH检查也返回`Permission denied (publickey)`。用户提醒后检查了非默认密钥，找到了既有`id_ed25519_github`并成功认证。此前“需要重新配置认证”的判断不完整；实际无需用户重新配置。

本仓库已配置正确的现有认证路径，后续普通同步命令为：

```bash
git -C /home/liulu/projects/VAR_COMM/publish/var-comm push -u origin main
```

后续推送前仍应检查远端是否新增提交；不能强推覆盖用户后来添加的文件。

本次仅发布准备，没有新训练、性能试验或holdout访问。既有固定m7混合收口结论不变。
