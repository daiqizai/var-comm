"""Report closed ARPC results. No model, channel, scoring or resampling calls."""
import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys

E = pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
sys.path.insert(0, str(E))
from a1_owner import O, desc, verify, put

METHODS = {
    'EC_VAR_PARTIAL': 'NeST-Com (entropy-coded)',
    'ARPC_ADAPTED': 'ARPC + LDPC–QAM (adapted)',
    'P100000': 'Latent-space JSCC (100k)',
    'Swin200000': 'SwinJSCC (adapted, 200k)',
    'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024': 'Adaptive BPG + LDPC',
}
METRICS = [
    ('psnr_db', 'PSNR (dB)', 1, 'higher'),
    ('lpips_alex', 'LPIPS-Alex', 1, 'lower'),
    ('dists', 'DISTS', 1, 'lower'),
    ('dreamsim', 'DreamSim', 1, 'lower'),
    ('dinov2_vitl14_cosine', 'DINO-L cosine', 1, 'higher'),
    ('convnext_top1_source_prediction', 'Prediction agreement (%)', 100, 'higher'),
    ('ms_ssim', 'MS-SSIM', 1, 'higher'),
]
GATES = {
    'a2b_source_v1': 'SOURCE1000_CLOSED',
    'a2b_link_v1': 'PHY18000_CLOSED',
    'a2b_rx_v1': 'ACTUAL_RX_CLOSED',
    'a2b_metrics_v1': 'CAL1000_FROZEN_CLOSED',
    'a3_source_v1': 'EVAL100_SOURCE_CLOSED',
    'a3_link_v1': 'EVAL900_PHY_CLOSED',
    'a3_rx_v1': 'EVAL_ACTUAL_RX_CLOSED',
    'a3_metrics_v1': 'EVAL100_FINAL_ANALYSIS_CLOSED',
}


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def count_text(rows, key):
    return '; '.join(f'{k}: {v}' for k, v in sorted(collections.Counter(str(r[key]) for r in rows).items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-final-closure', required=True)
    parser.add_argument('--expected-visual-review', required=True)
    args = parser.parse_args()
    final = O / 'a3_metrics_v1'
    assert desc(final / 'root_closure_review.json')['sha256'] == args.expected_final_closure
    inputs = []
    closures = {}
    for stage, status in GATES.items():
        path = O / stage / 'root_closure_review.json'
        review = read(path)
        assert review['status'] == status and review['all_processes_gone'] and review['unresolved'] == 0
        for pin in review['controls'] + review['outputs']:
            verify(pin)
        closures[stage] = review
        inputs.append(desc(path))
    figures = O / 'final_figures_v1'
    visual_path = figures / 'actual_visual_review.json'
    assert desc(visual_path)['sha256'] == args.expected_visual_review
    visual = read(visual_path)
    assert visual['status'] == 'PASS' and visual['actual_opened_both_pngs'] is True
    assert visual['figure_manifest'] == desc(figures / 'manifest.json')
    assert visual['reviewer']
    for pin in read(figures / 'manifest.json')['files']:
        verify(pin)
    assert {pathlib.Path(p['path']).name for p in visual['pngs']} == {
        'matched100_seven_metrics.png', 'matched100_paired_differences.png'}
    for pin in visual['pngs']:
        verify(pin)
    inputs += [desc(visual_path), desc(figures / 'manifest.json')]

    summary = csv_rows(final / 'matched100_summary.csv')
    paired = csv_rows(final / 'nest_minus_arpc_paired.csv')
    assert len(summary) == 105 and len(paired) == 21
    assert {r['method'] for r in summary} == set(METHODS)
    assert all(int(r['source_count']) == 100 for r in summary + paired)
    policies = read(O / 'a2b_metrics_v1/frozen_policies.json')
    assert policies['status'] == 'FROZEN_FROM_FULL_CAL1000'
    assert set(policies['policies']) == {'4', '10', '19'}
    frame_rows = read(O / 'a3_rx_v1/logical_receivers.json')
    assert len(frame_rows) == 900
    assert len({(r['source_id'], r['snr_db'], r['noise_seed']) for r in frame_rows}) == 900
    lengths = csv_rows(O / 'a3_source_v1/source_lengths.csv')
    assert len(lengths) == 700
    for rel in ['a2b_metrics_v1/frozen_policies.json', 'a3_rx_v1/logical_receivers.json',
                'a3_source_v1/source_lengths.csv', 'a3_metrics_v1/matched100_summary.csv',
                'a3_metrics_v1/nest_minus_arpc_paired.csv', 'a3_metrics_v1/bootstrap_identity.json']:
        inputs.append(desc(O / rel))

    lines = [
        '# ARPC：固定 100 源匹配比较', '',
        '本轮完成官方适配权重的独立接收资格、有限候选的完整 cal1000 校准，以及冻结策略后的 '
        'external_common100_v1 测试。测试共 900 个逻辑方法帧：100 源 × 4/10/19 dB × 每源三次噪声。'
        '所有失败均计入分母；比较单位是先平均三次噪声的来源。', '',
        '五种方法使用相同的 100 个 source ID。NeST-Com 19 dB 采用 T1-C m10 修正版；'
        'P100k、Swin200k 和 BPG 的既有逐源分数直接复用。此结果不代表整个 common500，'
        '也不称为新的盲测。', '',
        '## 冻结策略与实际接收', '',
        '每个 SNR 在两项预先提名的完整策略中，按 cal1000 的来源平均 DINO-L 选择；'
        'PSNR/LPIPS 不参与选取，测试结果不反馈选模。各源仍执行实际码长回退。', '',
        '| SNR (dB) | 冻结策略 | 发送实际终点分布 | 接收状态 | 灰图帧 / 300 |',
        '|---:|---|---|---|---:|',
    ]
    for snr in [4, 10, 19]:
        rows = [r for r in frame_rows if int(r['snr_db']) == snr]
        assert len(rows) == 300
        lines.append(f"| {snr} | {policies['policies'][str(snr)]} | {count_text(rows, 'endpoint')} | "
                     f"{count_text(rows, 'actual_status')} | {sum(r['image'] is None for r in rows)} / 300 |")
    wrong = sum(bool(r['accepted_wrong_information']) for r in frame_rows)
    lines += ['', f'实际接收后审计发现 {wrong} 帧存在合法接收但信息不等于发送端的情况；'
              '真值只用于固定输出后的审计，没有纠正接收数据。原始逐帧文件同时保存发送终点、'
              '实际译出终点、接收状态和来源解析结果。', '', '## 七指标完整结果', '',
              '均值及逐点 95% 来源 bootstrap 区间如下；只读取已经生成的统计结果。'
              '区间未做多重比较校正，不据此声称所有条件总体最优。']
    for metric, label, scale, direction in METRICS:
        lines += ['', f'### {label}（{"越高越好" if direction == "higher" else "越低越好"}）', '',
                  '| 方法 | 4 dB | 10 dB | 19 dB |', '|---|---:|---:|---:|']
        for method, name in METHODS.items():
            cells = []
            for snr in [4, 10, 19]:
                items = [r for r in summary if r['method'] == method and r['metric'] == metric and int(r['snr_db']) == snr]
                assert len(items) == 1
                r = items[0]
                cells.append(f"{float(r['mean'])*scale:.4f} [{float(r['ci_low'])*scale:.4f}, {float(r['ci_high'])*scale:.4f}]")
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 直接来源配对差：NeST-Com − ARPC', '',
              '差值在每源层面形成后重采样，未相减两条边际区间。预测一致率差的单位是百分点；'
              '其余指标使用原量纲。零值、负值和跨零区间完整保留。', '',
              '| 指标 | SNR (dB) | 配对均值差 [95% 区间] | 该逐点区间的方向 |', '|---|---:|---:|---|']
    for metric, label, scale, direction in METRICS:
        for snr in [4, 10, 19]:
            r = next(r for r in paired if r['metric'] == metric and int(r['snr_db']) == snr)
            lo, hi = float(r['ci_low']), float(r['ci_high'])
            if lo <= 0 <= hi:
                finding = '包含零，方向不明确'
            else:
                favorable = (lo > 0) if direction == 'higher' else (hi < 0)
                finding = '有利于 NeST-Com' if favorable else '有利于 ARPC'
            lines.append(f"| {label} | {snr} | {float(r['mean'])*scale:.6f} [{lo*scale:.6f}, {hi*scale:.6f}] | {finding} |")
    lines += ['', '## 源码与成本范围', '',
              'N=1024 个复符号，68 头部 + 956 正文，标称 Es=2。caption、caption 长度、'
              '视觉流、模式/边界/终止和对齐包含在实际源码载荷中；正文另外支付 13 bit 长度和 '
              '16 bit CRC。profile 经付费头部传输；不能装下时仍保留完整空帧及失败分母。', '',
              '| 终点 | 测试 100 源载荷 bit：最小 / 均值 / 最大 |', '|---:|---:|']
    for endpoint in range(1, 8):
        vals = [int(r['payload_bits']) for r in lengths if int(r['endpoint']) == endpoint]
        assert len(vals) == 100
        lines.append(f'| {endpoint} | {min(vals)} / {statistics.mean(vals):.2f} / {max(vals)} |')
    lines += ['', 'ARPC 使用原官方适配先验、BLIP caption、接收端独立文本编码与原采样配置。'
              'native TX/RX 在 leo 的已登记 GPU 与独立环境执行；评分使用原 H800 指标环境。'
              '原 mixed arithmetic、FP32/TF32 设置和权重身份均随登记披露。'
              '阶段墙时用于资源规划，含 I/O、初始化等开销，不能与旧共享 H800 计时计算速度倍率。', '',
              '## 执行、限制与材料', '',
              'A1 的初始 CUDA/NVML 身份失败以及工程 PHY 接收环境失败均保留；对应独立修订只处理'
              '身份或解释器读取范围，成功科学调用未重复。各阶段实际退出、失败记录、完整账本核验和'
              '输入输出 SHA 随轻量发布提供；合成测试不计作模型实验完成。', '',
              '本轮没有训练、新基线重评分、HiFi 恢复、额外功率/分层方案或完整 common500 ARPC 扩展。'
              '来源 bootstrap 仅生成一次：10,000 次重采样、seed 2026100701。'
              '其不覆盖训练随机性、历史选题和全部架构不确定性；历史基线噪声 seed 不完全相同，'
              '此处是来源配对。', '',
              '图表在 ../final_figures_v1/，完整均值、配对区间与逐源记录在 ../a3_metrics_v1/。'
              '恢复备份依用户要求延期。正式交付须另核正常 Git 发布和远端 SHA。', '']
    out = O / 'final_report_v1'
    assert not out.exists()
    out.mkdir()
    (out / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf8')
    put(out / 'manifest.json', {'status': 'CLOSED_RESULTS_REPORTED_PENDING_PUBLICATION',
        'source': desc(pathlib.Path(__file__)), 'inputs': inputs,
        'outputs': [desc(out / 'REPORT_zh.md')], 'new_model_calls': 0,
        'new_channel_calls': 0, 'new_scores': 0, 'new_bootstrap_draws': 0})
    print(json.dumps(desc(out / 'manifest.json')))


if __name__ == '__main__':
    main()
