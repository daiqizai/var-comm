"""Build a report-level archive, preserving mixed-purpose branches and PFS assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


V2 = "experiments/swinjscc-mismatch-zpac-20261120/"
DIR3 = "experiments/dir3-mimo-holorate-20261121/FINAL_REPORT_DIR3_MIMO_HOLORATE.md"
DIR4 = "experiments/dir4-rf1bit-noma-20261122/FINAL_REPORT_DIR4_RF1BIT_NOMA.md"
DIR5 = "experiments/dir5-otfs-genvar-20261122/FINAL_REPORT_DIR5_OTFS_GENVAR.md"
DIR6 = "experiments/dir6-matched-breakthroughs-20261123/FINAL_REPORT_DIR6_MATCHED_BREAKTHROUGHS.md"
DIR7 = "experiments/dir7-fas-persist-impulse-20261124/FINAL_REPORT_DIR7_FAS_PERSIST_IMPULSE.md"
MAIN = "582038a6cb5fd0e936e93748ac7df959daee74e3"
FOLLOW = "reports/direction_followups_20261123/README_zh.md"
ROOT = Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
CKPTS = ROOT / "liberai-checkpoint/liulu/var_comm"
RUNS = ROOT / "liberai-tmp/liulu/liulu-var-comm"
# Family, title, Git revision, report, PFS run folder, scientific boundary.
ROWS = [
    ("4B", "三用户过载接收", "ec608c7d89d78704b4efa7d32004fafec85f6766", DIR4, "dir4_rf1bit_noma_v1", "相同门控替换干扰副本，检验弱用户收益。"),
    ("6C", "四用户过载接收", "faaa2799f67593f1fe7846a4b421701a19d81f4d", DIR6, "dir6_matched_breakthroughs_v1", "四用户、两接收天线。原两组近远效应和固定检查点归入当前主线。"),
    ("5A", "高多普勒下波形与训练", "316d82b035b7af8acf3ce2ec95c34c0a0a4df544", DIR5, "dir5_otfs_genvar_v1", "平稳高多普勒配平训练增益 0.39/0.52 dB。"),
    ("6A", "连续深遮挡的二维扩频", "faaa2799f67593f1fe7846a4b421701a19d81f4d", DIR6, "dir6_matched_breakthroughs_v1", "连续遮挡 16/48 时为 +5.18 dB。其他等资源酉扩频仍是待比对象。"),
    ("7A", "六锚点流体天线选端", "db4e5865d9010d5f5b0866400861f0bc781fdea4", DIR7, "dir7_fas_persist_impulse_v1", "0 dB 平均与 p10 分别 +3.75/+5.48 dB。相同观测直接选择最佳锚点是待比对象。"),
    ("2 V1", "单层失配校准", "f5bb7c4dd06f76457912cf102701b09b4c62bb57", V2+"FINAL_REPORT.md", "dir2_swinjscc_mismatch_v1", "保留为多尺度收发校准的前身。"),
    ("2 V2", "过时反馈下的收发分工", "7a2d1dec5cc19b958c9ccc68d4894cb8d82595b9", V2+"FINAL_REPORT_V2.md", "dir2_swinjscc_mismatch_v2", "10 dB 原联合校准比冻结高 2.93 dB。反馈载荷单独计费。"),
    ("2 V3", "带内参考", "f7d764847ba3abd6e6fb4102f04b89e9b1412611", V2+"FINAL_REPORT_V3_BLIND.md", "dir2_blind_csifree_v3", "连续波形中恢复信道、噪声和源幅度。"),
    ("2 V4", "有限星座与广播", "65a5a8e14b92eb7fad92eb6e6444418e52d9a073", V2+"FINAL_REPORT_V4_UNIVERSAL_BLIND.md", "dir2_universal_blind_v4", "固定噪声底的 64-QAM 结果与显式导频资源核算一起保存。"),
    ("3A", "多天线带内参考", "165b6f200e38f37766b499111f3f9c419aa0b086", DIR3, "dir3_mimo_holorate_v1", "2×2 与 4×4 属于空间复用扩展。"),
    ("3B", "丢包与提前终止", "165b6f200e38f37766b499111f3f9c419aa0b086", DIR3, "dir3_mimo_holorate_v1", "保留分组布局的增益以及块内 Hadamard 的不利结果。"),
    ("4A", "低比特无线前端", "ec608c7d89d78704b4efa7d32004fafec85f6766", DIR4, "dir4_rf1bit_noma_v1", "保留固定门限与因果增益控制的匹配结果。"),
    ("5B", "连续特征与码本修正", "316d82b035b7af8acf3ce2ec95c34c0a0a4df544", DIR5, "dir5_otfs_genvar_v1", "早期码本修正的独立收益很小。"),
    ("6B", "极窄带 VAR 补全", "faaa2799f67593f1fe7846a4b421701a19d81f4d", DIR6, "dir6_matched_breakthroughs_v1", "同观测开启先验时的感知收益和 PSNR 代价一起保存。"),
    ("7B", "数字分层与延迟反馈", "db4e5865d9010d5f5b0866400861f0bc781fdea4", DIR7, "dir7_fas_persist_impulse_v1", "保留基于误块率抽样的可靠性与保真度取舍。"),
    ("7C", "脉冲干扰抑制", "db4e5865d9010d5f5b0866400861f0bc781fdea4", DIR7, "dir7_fas_persist_impulse_v1", "中等脉冲下有优势，较强且更密集时经典置零反超。"),
    ("1", "数字叠加广播", "1431eccf3aeac243ca18d2b016937c97ce4369b7", "results/sp_nest_probe_v1/REPORT_zh.md", "sp_nest_probe_v1", "保留 Pilot50 的正负结果及 SP2 未启动的原始判决。"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repository", type=Path)
    p.add_argument("destination", type=Path)
    cfg = p.parse_args()
    rows = []
    lines = ["# 图像传输研究方向归档", "", "2026 年 11 月 24 日。本轮资源集中于 4B＋6C。"
             "每个条目按具体路线归类。同一 Git 分支上的其他方法可独立归档。", "",
             "| 状态 | 方向 | 结果与边界 | 原始报告 |", "|---|---|---|---|"]
    for key, title, rev, report, folder, note in ROWS:
        cmd = ["git", "-C", str(cfg.repository)]
        raw = subprocess.check_output(cmd+["show", f"{rev}:{report}"])
        branch_tip = subprocess.check_output(cmd+["for-each-ref", "--points-at", rev, "--format=%(refname:short)", "refs/heads"], text=True).splitlines()
        state = "active" if key in ("4B", "6C") else "archived"
        rows.append(dict(direction=key, title=title, status=state, commit=rev, known_local_branch=branch_tip,
                         report=report, report_sha256=hashlib.sha256(raw).hexdigest(), scope=note,
                         checkpoint_root=str(CKPTS/folder) if (CKPTS/folder).is_dir() else None,
                         original_run_root=str(RUNS/folder) if (RUNS/folder).is_dir() else None))
        link = f"https://github.com/daiqizai/var-comm/blob/{rev}/{report}"
        status = "当前" if state == "active" else "归档"
        lines.append(f"| {status} | {key} · {title} | {note} | [原始记录]({link}) |")
    lines.extend(["", f"较早的逐部件配对复核与负结果见[主分支补充对照](https://github.com/daiqizai/var-comm/blob/{MAIN}/{FOLLOW})。"
                  "各研究提交、报告哈希及现存 PFS 权重和运行目录记录于 [archive_manifest.json](archive_manifest.json)。"
                  "旧 NeST 稿件和此前暂停的四个训练臂保留其项目记录。", ""])
    cfg.destination.mkdir(parents=True, exist_ok=True)
    with (cfg.destination/"archive_manifest.json").open("x", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (cfg.destination/"ARCHIVE_zh.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
