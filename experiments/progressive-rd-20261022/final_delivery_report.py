"""CPU-only report of closed receipts; no model, metric or bootstrap calls."""
from common import *
import csv
import collections
import shutil


def main():
    d = O / 'cache_expansion_delivery_v1'
    closure = read(d / 'closure_audit.json')
    assert closure['status'] == 'PASS'
    wait = read(L / 'cache_expansion_closure_dispatch_v1/actual_wait.json')
    assert type(wait['returncode']) is int and wait['returncode'] == 0 and not wait['timed_out']
    assert process(wait['process']['pid']) is None
    launch = read(L / 'cache_expansion_closure_dispatch_v1/launch.json')
    assert process(launch['parent_process']['pid']) is None
    assert process(read(L / 'expansion_initial_control_v1/launch.json')['process']['pid']) is None
    root = read(L / 'expansion_dispatch_v1/completion.json')
    assert root['all_exit0'] is True
    index = read(checked(closure['source_index']))
    seed = read(checked(index['seed_index']))
    reg = read(checked(closure['registration']))
    for sh in closure['shards']:
        w = read(checked(sh['actual_wait']))
        assert type(w['returncode']) is int and w['returncode'] == 0
        assert not any(w[k] for k in ['terminated', 'killed', 'timed_out'])
        assert process(w['process']['pid']) is None
    out = O / 'final_delivery_v1'
    assert not out.exists()

    # Descriptor sizes were verified in the completed seed/expansion audits.
    # No repeat traversal of native arrays or model state is required here.
    payload = collections.defaultdict(dict)
    for row in seed['sources'] + index['new_sources']:
        for kind in ['latents', 'wire', 'received']:
            item = row[kind]
            p = pathlib.Path(item['path'])
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == item['bytes']
            old = payload[kind].setdefault(item['path'], item)
            assert old == item
    storage = {k: dict(files=len(v), bytes=sum(x['bytes'] for x in v.values())) for k, v in payload.items()}
    checkpoints = read(O / 'decoder_delivery_v1/checkpoint_manifest.json')
    for c in checkpoints:
        checked(c['completion']); checked(c['actual_wait']); checked(c['calibration'])
        p = pathlib.Path(c['checkpoint']['path'])
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == c['checkpoint']['bytes']

    starts = []; ends = []; throughput = []
    for sh in closure['shards']:
        j = sh['shard']; w = read(checked(sh['actual_wait']))
        launch = read(L / 'expansion_dispatch_v1' / f'shard_{j:02d}' / 'launch.json')
        starts.append(launch['started_unix']); ends.append(launch['started_unix'] + w['elapsed_seconds'])
        throughput.append(dict(shard=j, sources=sh['sources'], endpoint_rows=8*sh['sources'],
            actual_wait_seconds=w['elapsed_seconds'], source_seconds=sh['source_seconds'],
            io_seconds=sh['io_seconds'], measured_io_fraction=sh['io_seconds']/sh['elapsed_seconds'],
            sources_per_second=sh['sources']/sh['elapsed_seconds'],
            endpoint_rows_per_second=8*sh['sources']/sh['elapsed_seconds'],
            peak_allocated_bytes=sh['peak_allocated_bytes']))
    makespan = max(ends)-min(starts)
    assert max(ends) < reg['absolute_deadline_unix']
    total_calls = {k: sum(x['counts'].get(k, 0) for x in closure['shards'])
                   for k in ['tx_probability_pass','independent_source_decode','predict_overwrite_advance','lpips_distance','ms_ssim']}
    assert total_calls['tx_probability_pass'] == 18968
    assert all(v == 151744 for k,v in total_calls.items() if k != 'tx_probability_pass')
    evidence = dict(status='CLOSED', observed_unix=time.time(), expansion_closure=pin(d/'closure_audit.json'),
        closure_actual_wait=pin(L/'cache_expansion_closure_dispatch_v1/actual_wait.json'),
        root_completion=pin(L/'expansion_dispatch_v1/completion.json'),
        decoder_closure=pin(O/'decoder_delivery_v1/closure_audit.json'),
        decoder_visual_review=pin(O/'decoder_delivery_v1/visual_review.json'),
        decoder_checkpoints=pin(O/'decoder_delivery_v1/checkpoint_manifest.json'),
        expansion_calls=total_calls, cache_sources=20068, full_endpoint_train_sources=19968,
        calibration_sources=100, endpoint_rows=160544, separate_stage0_sources=32,
        separate_stage0_endpoint_rows=64, unique_source_TX_including_stage0=20100,
        payload_storage=storage, payload_storage_excludes='logs, metadata, checkpoints, old RGB/token caches and stage0',
        expansion_makespan_seconds=makespan, expansion_sources_per_second=18968/makespan,
        latest_worker_exit_estimated_unix=max(ends), absolute_deadline_unix=reg['absolute_deadline_unix'],
        exit_time_basis='launch started_unix plus real parent wait elapsed; control timestamps, not device timing',
        new_model_metric_PHY_or_bootstrap_calls_in_this_report=0, backups_deferred=True)
    out.mkdir()
    receipts = out / 'execution_receipts'
    receipts.mkdir()
    copies = []
    originals = [(L/'expansion_initial_control_v1/launch.json', 'owner_launch.json'),
                 (L/'expansion_dispatch_v1/completion.json', 'all_shards_terminal.json')]
    for i in range(8):
        for name in ['launch.json', 'actual_wait.json', 'stdout.log', 'stderr.log']:
            originals.append((L/'expansion_dispatch_v1'/f'shard_{i:02d}'/name, f'shard_{i:02d}_{name}'.replace('.log','.txt')))
        originals.append((L/'expansion_cache_v1'/f'shard_{i:02d}'/'completion.json', f'shard_{i:02d}_completion.json'))
    for src, name in originals:
        before = pin(src); dst = receipts/name
        assert before['bytes'] < 10_000_000 and not dst.exists()
        shutil.copyfile(src, dst)
        assert pin(src) == before and sha(dst) == before['sha256']
        copies.append(dict(source=before, copy=pin(dst)))
    save(receipts/'copy_manifest.json', copies)
    save(out/'completion_evidence.json', evidence)
    save(out/'recovery_manifest.json', dict(cache_index=closure['source_index'], seed_index=index['seed_index'],
        registration=closure['registration'], compact_registration=closure['registration_compact'],
        exact_registration_restore=pin(E/'restore_expansion_registration.py'),
        checkpoints=checkpoints, native_environment=str(PY),
        latent_precision='FP32', endpoint_order=reg['endpoints'],
        resume_execution_tested=False, validated_resume_cli_available=False,
        limitations=['Checkpoint state is complete, but no resumed update or cross-GPU recovery has been executed.',
                    'Current training entry is one-shot; a reviewed resume entry is needed before any future update.',
                    'Original RGB/tokens and frozen weights remain dependencies; this is not a standalone backup.']))
    with (out/'throughput.csv').open('x', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(throughput[0]));w.writeheader();w.writerows(throughput)
    means=list(csv.DictReader((d/'endpoint_means.csv').open()))
    lines=['# 渐进 token 率失真研究：本轮完成交付','',
        '本轮登记的缓存扩展、三组匹配 decoder 试训与最终校准均已完成。八个扩展子进程均由父进程实际 wait 得到退出码 0，无超时/终止；原调度进程已退出。全部已登记调用保持原预算，没有追加训练或信道实验。','',
        '## 数据覆盖','',
        '- 新扩展 18,968 个训练源 × 8 个端点 = 151,744 条；与已闭合种子库合并为 19,968 个训练源和 100 个校准源，各有 8 个端点，共 160,544 条。',
        '- 另有 benchmark 的 32 个训练源 × 2 个端点 = 64 条，单独保留。没有补跑其余 6 个端点；不能写成完整 20,000 × 8 库。',
        '- 包括 benchmark，独特 TX 源总数 20,100；复用原 RGB/token，不重做 tokenizer。每源一次概率遍历生成独立终止源码流。',
        '- 扩展的源解码、原生 predict–overwrite–advance、LPIPS 和 MS-SSIM 各 151,744 次。模型/指标加载另列于 closure ledger，不计为图像恢复。',
        '- 保存实际码流、接收 token、FP32 完成 latent、完整精度码长及质量。除已登记预览，不保存全量高分辨率重建图。无无线 PHY；这些是无损源码接收条件下的率失真标签。','',
        '## 训练源的描述性率失真表','',
        '以下为冻结当前 Dc 的 19,968 训练源均值；不是新训练模型的测试集性能。码长包含实际算术终止，未计无线保护和帧开销。校准100的独立均值保留在同一 CSV，不混合人口。','',
        '| (m,K) | 终止 bit | bit/pixel | PSNR (dB) | LPIPS | MS-SSIM |',
        '|---|---:|---:|---:|---:|---:|']
    for r in means:
        if r['split']=='train':
            lines.append(f"| ({r['m']},{r['K']}) | {float(r['terminated_bits']):.3f} | {float(r['bits_per_pixel']):.6f} | {float(r['psnr_db']):.4f} | {float(r['lpips_alex']):.6f} | {float(r['ms_ssim']):.6f} |")
    lines += ['', '## 匹配试训结论与限制','',
        '三组各 5,000 更新、batch4、20,000 样本暴露，相同初始化与数据顺序；仅 decoder 更新。多前缀无条件相对等更新旧配方，在 (7,0)/(7,25) 的 PSNR 约提高 0.337/0.319 dB，在完整 (10,0) 下降约 0.152 dB。较长端点 LPIPS 多数变差。端点条件带来的 PSNR 增量约 0.002–0.028 dB，LPIPS 多数配对区间跨零。',
        '相对冻结当前 Dc，多前缀无条件在 (7,0) 提高约 0.307 dB，在完整端点下降约 0.270 dB。没有全端点一致获益，不能直接替换当前 WCL 模型。所有端点、五个对比和不利结果均保留在 decoder_delivery_v1。',
        '这只是一个训练随机种子的 calibration100 结果；一次已登记来源配对 bootstrap，逐点区间未校正多重比较，不代表训练种子不确定性、盲测或无线收益。缓存扩展没有重做该统计。','',
        '## 实际吞吐与存储','',
        f'扩展八分片从最早启动到最后实际退出约 {makespan:.3f} 秒，聚合 {18968/makespan:.4f} 源/秒（含前三卡等待训练造成的启动错开）。全体在原绝对截止时间前退出；截止时间没有重置。',
        '逐分片数据见 throughput.csv。I/O 秒数是已埋点文件阶段，不能当作全机器磁盘等待，也不能据此断言无 PFS/CPU 瓶颈；实测源处理含 CPU 算术码及模型工作。设备为驱动标识 NVIDIA RPBZZZ6，FP32/TF32 off，原6 math/2 interop，独立单卡推理，没有 DDP 或跨卡显存池。','',
        '| 数据类型（种子＋扩展） | 文件数 | 实际文件字节 |','|---|---:|---:|']
    for k,v in storage.items():lines.append(f"| {k} | {v['files']} | {v['bytes']} |")
    lines += ['', '以上按已校验的实际描述符汇总，不含元数据、日志、checkpoint、原 RGB/token 依赖和 benchmark。FP32 latent 的真实体积不同于最初 FP16 容量估算。大文件继续放在共享 PFS，未下载或新做完整备份。','',
        '## 后续恢复清单','',
        '- recovery_manifest.json 给出缓存索引、实际注册、精确恢复注册的源码与三个最终 checkpoint 路径/字节/SHA。cache_expansion_index_v1.json 可按 source_id 找到扩展 completion、wire、received 和 latent；其 seed_index 引用同样定位原种子库。',
        '- 全量源身份及预处理依赖按 expansion_registration 的 selected_sources 对齐；保留 source_index，不重新编号。加载前核验 SHA、source_id、端点顺序和 FP32 latent 身份。',
        '- 三个 step_005000.pt 含 model/optimizer/scheduler/scaler=None、global_step、processed_samples、Python/NumPy/Torch/CUDA RNG、mixture RNG、data_order、config。checkpoint 在最终校准前保存，内嵌 calibration=None；必须同时保留 completion.json 绑定的最终 calibration.csv。',
        '- 尚未执行单卡恢复后的更新，也没有已验证的 resume CLI；原训练入口为一次性入口，不能直接重跑冒充续训。后续应先实现受审的状态恢复入口，不能只加载模型权重后丢弃 optimizer/RNG/order。迁移设备或改变 batch/world size 不承诺逐位重现。',
        '- 依赖当前冻结 VAE、VAR、Dc、原 RGB/token 缓存和原环境。本清单不是独立恢复包；用户允许推迟备份。','',
        '## 已保留失败与未开展范围','',
        '原种子 v1 的8次导入失败发生在模型加载主体之前，零成功科学调用；原现场和失败 reservation 全部保留。修复只处理导入命名冲突。早期绘图裁切及布局守卫误检也保留，未因此重算科学结果。',
        '本轮未开展 encoder/VAR/codebook 更新、码率代理训练、内容选择器、分层 PHY、heldout/common500、新超参或随机种子扫描。临时八卡任务到此停止；进一步研究需新的有限方案，不自动消耗剩余窗口。','',
        '## 文件入口','',
        '- `../cache_expansion_delivery_v1/endpoint_m*_k*.csv`：8份逐源端点表，每份20,068行，源文件完整精度；`endpoint_means.csv` 分列训练与校准。',
        '- `../decoder_delivery_v1/REPORT_zh.md`、`summary.csv`、`paired_differences.csv`：全部匹配试训结果。',
        '- `../decoder_delivery_v1/figures_final_v1/calibration_rd.*` 与 `calibration_paired.*`：已实际检查的矢量曲线与配对差图，不重绘。',
        '- `completion_evidence.json`、`recovery_manifest.json`、`throughput.csv`：本轮闭合、计算范围与恢复依据。','']
    with (out/'REPORT_zh.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    save(out/'manifest.json', [pin(p) for p in sorted(out.rglob('*')) if p.is_file()])
    print(json.dumps(dict(report=pin(out/'REPORT_zh.md'),manifest=pin(out/'manifest.json'),scientific_calls=0)))


if __name__ == '__main__':
    main()
