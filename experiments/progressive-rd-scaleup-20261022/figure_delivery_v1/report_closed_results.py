"""Format already-closed results and recovery metadata; no new statistics."""
import argparse
import csv
from pathlib import Path

from plot_closed_results import (
    ENDS, REG_SHA, SEEDS, STYLES, checked, load_closed, pin, read, save_json,
)

METRIC_LABELS = [('psnr_db', 'PSNR ↑'), ('lpips_alex', 'LPIPS ↓'),
                 ('ms_ssim', 'MS-SSIM ↑'), ('mse', 'MSE ↓'), ('utility', 'MSE + 0.1 LPIPS ↓')]
CONTRASTS = [
    (a, 'current_dc') for a, *_ in STYLES[1:]
] + [
    ('multiprefix_unconditional', 'old_recipe_control'),
    ('multiprefix_full_anchor', 'old_recipe_control'),
    ('multiprefix_full_anchor', 'multiprefix_unconditional'),
]


def interval(row):
    return f"{float(row['mean']):+.6f} [{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}]"


def main(root, run):
    root, run = Path(root), Path(run)
    _, pairs, _, _ = load_closed(root, run)
    closure = read(root / 'delivery_v1/closure_audit.json')
    reg = read(root / 'registration.json')
    out = root / 'report_v1'
    assert not out.exists()

    # The completed CPU audit has already hashed these large artifacts. Only
    # existence/size is rechecked here; neither checkpoint loading nor replay.
    checkpoints, compute, receipts = [], [], []
    for job in closure['jobs']:
        completion_path = checked(job['completion'])
        complete = read(completion_path)
        wait_path = checked(job['wait'])
        wait = read(wait_path)
        for key in ['checkpoint', 'initial_checkpoint']:
            p = Path(job[key]['path'])
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == job[key]['bytes']
        entry = dict(job=job['job'], arm=job['arm'], seed=job['seed'],
                     final_checkpoint=job['checkpoint'], initial_checkpoint=job['initial_checkpoint'],
                     completion=job['completion'], actual_wait=job['wait'],
                     validation=complete['validation'], ledger=complete['ledger'],
                     training_updates=complete['updates'], identity=job['identity'])
        checkpoints.append(entry)
        compute.append(dict(job=job['job'], arm=job['arm'], seed=job['seed'],
                            updates=complete['training_updates'], main_samples=complete['training_samples'],
                            anchor_samples=complete['anchor_training_samples'],
                            teacher_forwards=complete['teacher_training_forwards'],
                            full_student_forwards=complete['full_student_training_forwards'],
                            validation_rows=complete['validation_endpoint_rows'],
                            unique_training_sources=job['unique_training_sources'],
                            unique_source_endpoint_pairs=job['unique_source_endpoint_pairs'],
                            training_seconds=job['training_seconds'],
                            worker_total_seconds=job['total_seconds'],
                            actual_wait_seconds=wait['elapsed_seconds'],
                            peak_allocated_bytes=job['peak_allocated_bytes']))
        receipts.extend([job['completion'], job['wait']])
    assert len(compute) == 6 and sum(x['updates'] for x in compute) == 60000
    assert sum(x['main_samples'] for x in compute) == 240000
    assert sum(x['anchor_samples'] for x in compute) == 80000
    assert sum(x['validation_rows'] for x in compute) == 9600

    out.mkdir()
    with (out / 'actual_compute.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compute[0]))
        writer.writeheader(); writer.writerows(compute)
    save_json(out / 'recovery_manifest.json', dict(
        registration=pin(root / 'registration.json'), registration_sha256=REG_SHA,
        parent_recovery_manifest=reg['parent_recovery_manifest'],
        training_cache_index=reg['train_index'],
        validation_cache_closure=closure['validation_closure'],
        frozen_model_routes=reg['model_routes'], frozen_model_states=reg['model_states'],
        bound_sources=reg['sources'], execution_lineage=closure['execution_lineage'],
        checkpoints=checkpoints, bootstrap=closure['bootstrap'],
        source_order=closure['source_order'], endpoints=reg['endpoints'],
        checkpoint_fields_from_frozen_serializer=[
            'model', 'optimizer', 'scheduler', 'scaler=None', 'global_step',
            'processed_samples', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng',
            'data_order', 'mixture_rng', 'config', 'registration_sha256',
            'arm', 'job', 'seed', 'teacher_sha256', 'calibration=None', 'reason',
        ],
        checkpoint_field_basis='Frozen train.py serializer; final checkpoint saved before final validation.',
        checkpoints_sha_verified_by=pin(root / 'delivery_v1/closure_audit.json'),
        checkpoint_loaded_in_this_report=False, actual_resume_tested=False,
        validated_resume_cli_available=False, backup_created=False,
        limitations=[
            'Keep the separately bound validation.csv; final checkpoint calibration is None.',
            'No resumed update or cross-device recovery has been executed.',
            'The original one-shot worker is not a resume command.',
            'Future continuation must restore optimizer, scheduler, RNG and exact data order.',
            'Original RGB/token/latent caches, frozen weights and native environment remain dependencies.',
            'This inventory is not a standalone backup; local backups remain deferred.',
        ],
    ))

    lines = ['# Decoder 扩展：实际计算与完整结果', '',
        '三种 decoder 训练设置、两个预定种子全部完成各 10,000 次更新及最终 200 源×8 端点评价。六个科学子进程均真实退出 0，完整账本由一次 CPU 汇总审计。原调度器保留两个启动前资源拒绝，因此其最终退出码为 1；独立恢复仅补了那两个从未启动的科学子进程，未重复已成功调用。', '',
        '## 比较范围', '',
        '- 训练库复用 19,968 源×8 个已完成的真实源码接收 latent；训练源不重做 tokenizer、TX 概率遍历或 VAR 补全。每组 40,000 次主样本暴露，共 240,000 次，并非每个源/端点都被训练遍历一次。实际覆盖见 actual_compute.csv。',
        '- old-recipe control、多前缀无条件和多前缀＋完整端点约束三组采用相同初始模型、每种子相同 source/endpoint 抽取顺序。旧配方对照使用原混合输入，多前缀组使用收到前缀后的真实完成 latent。',
        '- 完整端点约束固定权重 1.0；每个约束组额外 10,000 次冻结教师前向和 10,000 次完整端点学生前向，各对应 40,000 个额外完整端点样本。没有扫描权重；等更新数/主样本暴露不代表相同 FLOPs。',
        '- 验证 200 源来自原 calibration1000，按预先固定哈希选择并排除旧 pilot100；与 decoder 训练源不重叠。这仍是历史校准人口，不是新盲测、common500 或无线质量证据。',
        '- 八端点全部保留，包括完整端点 (10,0)。码率为真实独立终止算术流 bit/pixel，未加入无线帧或 LDPC/QAM 开销。',
        '- 新验证缓存 200 次 TX 遍历、1600 次源码解码/VAR 补全及对应基准评分已完成；六模型最终评价合计 9600 条，冻结基准的 1600 条直接复用。没有 encoder、VAR、codebook 更新或 PHY 调用。', '',
        '## 统计定义', '',
        '只使用已登记的一次 10000×200 来源配对 bootstrap（seed=2026102209）。逐种子结果与两种子逐源均值的结果均报告；区间以源为重采样单位，不估计训练种子总体的不确定性。所有区间为逐点 95% CI，未做多重比较校正。utility 为 MSE+0.1 LPIPS，越低越好。配对差统一为第一个训练设置减第二个；PSNR/MS-SSIM 越高越好，LPIPS/MSE/utility 越低越好。', '',
        '两种子不能充分刻画训练稳定性；不根据本轮校准最优点自动替换 WCL 模型，也不将无损源码接收曲线当作固定无线预算收益。', '',
        '## 预先关注的短/完整端点', '',
        '以下只突出 (7,0) 与 (10,0) 便于回答短前缀收益和完整端点保持问题；八端点、全部六对比、两种子及均值结果在下方完整表及原始 CSV 中。数值仅显示舍入，原始 CSV 精度未改变。', '',
        '| 对比 | (m,K) | ΔPSNR [95% CI] | ΔLPIPS [95% CI] |',
        '|---|---|---:|---:|']
    for a, b in CONTRASTS:
        for m, k in [(7, 0), (10, 0)]:
            name = a + '_minus_' + b
            lines.append(f'| {a} − {b} | ({m},{k}) | {interval(pairs[name,"mean_of_2",m,k,"psnr_db"])} | {interval(pairs[name,"mean_of_2",m,k,"lpips_alex"])} |')
    lines += ['', '## 资源与恢复', '',
        '实际每组更新、样本暴露、覆盖数、耗时和峰值显存见 actual_compute.csv。本轮为共享 GPU 的质量研究；其他任务负载变化会影响耗时，不把这里的 wall time 当成公平延迟基准。绝对截止时间没有重置。',
        'recovery_manifest.json 给出各最终/初始 checkpoint、原始注册、缓存索引、冻结模型、精确源文件与实际退出链。checkpoint 文件 SHA 已在闭合审计核验；本报告不重新加载模型。文件由冻结序列化器保存 optimizer、scheduler、RNG 和 data_order，但实际恢复训练及跨设备恢复均未测试。最终 checkpoint 在评价前保存，须另保留绑定的 validation.csv。',
        '原始 metadata 注册失败、两次启动前拒绝及原 owner/supervisor 的非零退出全部保留。最终科学成功是原四组＋首次补启动两组，不伪称原 owner 全部退出 0。',
        '大文件只保留共享 PFS，备份按用户要求延后。没有自动追加训练种子、encoder/VAR/codebook 训练、无线网格或新超参数。', '',
        '## 文件对应', '',
        '- ../delivery_v1/source_metrics.csv：11200 条逐源端点记录。',
        '- ../delivery_v1/summary.csv：400 个均值与 CI；paired_differences.csv：720 个直接配对差与 CI。',
        '- ../delivery_v1/endpoint_rates.csv：八端点的实际平均终止码率。',
        '- all_paired_results.md：全部六个对比、两单独种子及两种子均值的五指标配对表。',
        '- ../figures_v1/：已准备绘图入口；只有真实渲染及独立 visual_review.json 完成后才能宣称图稿审查通过。',
        '- publication 仍需原仓库必要检查、正常提交/推送与远端 SHA 核验；本报告自身不是发布收据。', '']
    (out / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')

    full = ['# All saved source-paired comparisons', '',
            'First arm minus second arm. All values below are formatted from the sealed CSV; no new statistics. Source-bootstrap intervals do not estimate training-seed population uncertainty.', '']
    for seed in [*SEEDS, 'mean_of_2']:
        for a, b in CONTRASTS:
            name = a + '_minus_' + b
            full += [f'## {seed}: {a} − {b}', '',
                     '| (m,K) | ' + ' | '.join(label + ' [95% CI]' for _, label in METRIC_LABELS) + ' |',
                     '|---|' + '---:|' * len(METRIC_LABELS)]
            for m, k in ENDS:
                full.append(f'| ({m},{k}) | ' + ' | '.join(interval(pairs[name, seed, m, k, metric]) for metric, _ in METRIC_LABELS) + ' |')
            full.append('')
    (out / 'all_paired_results.md').write_text('\n'.join(full), encoding='utf-8')
    save_json(out / 'provenance.json', dict(status='CLOSED_STATISTICS_REPORT_ONLY',
        source=pin(Path(__file__)), statistics_manifest=pin(root / 'delivery_v1/manifest.json'),
        closure=pin(root / 'delivery_v1/closure_audit.json'),
        actual_wait=pin(run / 'summary_dispatch_v1/actual_wait.json'),
        new_scientific_calls=0, new_bootstrap=0, scientific_receipts=receipts))
    save_json(out / 'manifest.json', dict(files=[pin(p) for p in sorted(out.iterdir()) if p.is_file()]))
    print(out / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    main(args.results, args.run)
