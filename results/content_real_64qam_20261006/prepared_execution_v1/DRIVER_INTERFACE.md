# initial_true200 CPU正式入口（本地实现，尚未登记运行）

本入口读取最终冻结shortlist，不执行预筛或选模。只运行四个whole臂；每臂、每个13/19 dB保留1–2个已选策略，原200源、6101/6102/6103三个噪声。最大9600帧、19200次header/body调用，全部沿原H ledger的initial_true200分仓。partial候选不混入。

## CLI及资源

```text
python -B h_payload_driver.py --config CONFIG --stage worker --worker-index 0
python -B h_payload_driver.py --config CONFIG --stage worker --worker-index 1
python -B h_payload_driver.py --config CONFIG --stage merge
```

这些是接口说明，尚未执行。主控须先封存新revision，按模板补全实际路径/SHA并验证全部原字节。两worker分别处理偶数/奇数源序号，各100源；CPU affinity各2个独立核，nice15，线程2，CUDA不可见。merge只在两个worker完成且退出后由owner调用，不启动任何后续GPU作业。

## 登记与无环contract

`payload_config.template.json`包含原预算路径、原ledger路径、完整phase_limits、固定source200、source/S1各层completion、已资格catalogue/backend和最终prescreen。模板内占位符必须替换；源文件哈希不代表已经部署。

先产生独立engineering contract：其status为H_PAYLOAD_CPU_ENGINEERING_SEALED，绑定冻结core、完整协议、catalogue、最终shortlist及ENGINEERING_CHOICES的SHA；**不要放execution_registration_sha256**。随后execution registration绑定contract、config与所有实际源码/依赖和输入文件，allowed_stage_ids含initial_true200。运行时driver在内存副本中注入该登记文件原SHA，原contract不修改，因此不存在哈希环。

原预算文件SHA、branch、全部phase_limits和原ledger绑定逐项核验；不建立新账本。qualification/prescreen/source200/S1 receipt沿outputs原SHA核验，S1根receipt必须绑定export-assets receipt。预筛所用source_dir_completion必须与实际发送码流的source200 completion完全相同。

## 私有结果接口

worker目录为`OUT/worker_0`或`OUT/worker_1`：

- `traces/NNNN.json`：status=H_INITIAL_CPU_SOURCE_TRACES，包含source_id/index、registration_sha256、source_bindings和frames数组。frames保持core实际RX输出，不替换错收payload或decoded bits。
- `source_checkpoints/NNNN.json`：绑定对应trace SHA与帧数。
- `traces.journal.jsonl`：每一完整帧立即flush/fsync；未完成源也保留已得到的接收记录。
- `completion.json`：status=H_INITIAL_TRUE200_CPU_WORKER_COMPLETE，绑定全部trace/CP/journal、driver config SHA、预算SHA、最终shortlist SHA、原ledger.worker身份与逐事件审计。

merge在`OUT/completion.json`写H_INITIAL_TRUE200_CPU_RECEIVE_COMPLETE，绑定两个worker completion及全部输出，含200源覆盖、全部logical frames、实际能量统计、完整ledger snapshot和事件结果SHA审计。逻辑事件计数不是新增译码次数；实际计费只能从ledger取得。没有本轮图像评分，source_decode_complete=false、arithmetic_source_decode_complete=false、images_scored=false。

CPU完成凭证在进程退出前写出，不能凭该文件证明进程退出。因此merge receipt明确workers_exit_verified=false并列出worker_identities；后续GPU owner必须以独立退出凭证匹配PID/start_ticks/UID/argv并确认两个worker已退出。不得仅看到CPU completion就重叠启动视觉作业。

## 失败处理

worker有attempt或failure而没有合法完成凭证时拒绝隐式续跑。ledger RESERVED/FAILED不会被清理、退款或自动重试；需要独立恢复登记。STOP在帧前检查，若header已经开始，原receive_frame完成该帧的合法header/body路径再到安全边界。异常时记录failure并保留journal及原账本。merge拒绝任何缺失、额外、不完整、或与实际ledger结果SHA不符的事件。

这些trace含实际接收bit数组，属于后续接收器的私有输入，不应整份作为公开轻量报告上传。
