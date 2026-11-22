"""Generate readable tables from the actual completed paired results."""
import csv
import json
from pathlib import Path
from study import STUDY, save_once


def rows(p):
    with p.open(encoding="utf-8") as f: return list(csv.DictReader(f))


def one(data,**where):
    found=[v for v in data if all(v[k]==str(value) for k,value in where.items())]
    if len(found)!=1: raise ValueError((where,len(found)))
    return found[0]


def fmt(z,places=3,mean="mean"):
    core=f"{float(z[mean]):+.{places}f}"
    if z.get("ci95_lo","")=="": return core
    a,b=(float(z[k]) for k in ("ci95_lo","ci95_hi"))
    return f"{core} [{a:+.{places}f}, {b:+.{places}f}]"


def main():
    root=STUDY;d5=root/"dir5"
    lines=["# 实际完成的机制检验与两版正式结果","",
           "除明示为均值差的条目，括号均为直接配对的逐点 95% 区间。PSNR 越高越好，DISTS 和 LPIPS 越低越好。",
           "","## 3B · 固定通道分组与可靠度后，空间变换增加多少","",
           "两臂都在相同初始化和通道抽样下训练 5,000 步，各自用 256 张验证图选点。对照保留符号扰动、三通道一组的装包、功率、接收参考及逐组可靠度，只移去空间 Hadamard 及其逆。第三列是两份新权重的直接比较。最后一列给出新完整模型对历史 raster 权重的完整系统差值。","",
           "| 图像 | 收块条件 | 完整 − 同装包无变换，PSNR (dB) | 完整 − 同装包无变换，DISTS | 完整 − raster，PSNR (dB) |",
           "|---|---|---:|---:|---:|"]
    h=rows(root/"evaluation/holo/paired.csv")
    for split,name in (("common500","500 张"),("kodak24","Kodak24")):
        for c,cn in (("first8","前 8/32"),("first16","前 16/32"),("contiguous16","中间连续丢 16/32"),("full","全 32 块")):
            key=dict(split=split,condition=c,role="image",left="full")
            lines.append(f"| {name} | {cn} | {fmt(one(h,**key,right='identity',metric='psnr_db'))} | "
                         f"{fmt(one(h,**key,right='identity',metric='dists'),4)} | {fmt(one(h,**key,right='reference',metric='psnr_db'))} |")
    lines+=["","每个条件每图使用原有的三个噪声重复。历史 raster 同时改变装包和可靠度，且来自早期独立训练。理想整组擦除时，单个通道在整块丢失后仍然缺失。完整有噪链路上的差值可以反映其余调制、噪声、接收和共同训练的相互作用。[均值](evaluation/holo/means.csv) · [全部配对](evaluation/holo/paired.csv) · [历史参考重放](evaluation/holo/historical_replay.csv)",
            "","## 4B · 允许跨用户回写后，弱用户增加多少","",
            "两臂各有 1,702,213 个可训练参数，训练 4,500 步。相同的本用户潜变量监督、两轮先验、门控及初始化。完整臂额外用其他用户的估计波形更新当前观测及干扰方差，再做线性分离。对照臂只门控地更新本用户潜变量。两臂计算量有差异。","",
            "| 图像 | 用户与输入 SNR (dB) | 对象 | 完整 − 自身去噪，PSNR (dB) | 完整 − 自身去噪，DISTS | 完整 − 单级，PSNR (dB) |",
            "|---|---|---|---:|---:|---:|"]
    nr=rows(root/"evaluation/noma/paired.csv")
    for split,name in (("common500","500 张"),("kodak24","Kodak24")):
        for c,cs,role,rn in (("over3x2_13_8_4","3×2 · 13/8/4","User3_4dB","弱用户"),
                    ("over3x2_13_8_4","3×2 · 13/8/4","ALL_USERS_MEAN","三用户平均"),
                    ("nearfar_12_3","2×2 · 12/3","User2_3dB","弱用户"),
                    ("balanced_10_10","2×2 · 10/10","ALL_USERS_MEAN","两用户平均")):
            key=dict(split=split,condition=c,role=role,left="full")
            lines.append(f"| {name} | {cs} | {rn} | {fmt(one(nr,**key,right='own_denoise',metric='psnr_db'))} | "
                         f"{fmt(one(nr,**key,right='own_denoise',metric='dists'),4)} | {fmt(one(nr,**key,right='reference',metric='psnr_db'))} |")
    lines+=["","每个点使用实际接收的同一组图像和信道张量。用户由固定移位组成，所有用户均值先在同一组中求平均。区间反映这些固定随机权重、图像组合和三个噪声重复下的差异。[均值](evaluation/noma/means.csv) · [全部角色与配对](evaluation/noma/paired.csv) · [历史参考重放](evaluation/noma/historical_replay.csv)",
            "","## 5A · 发送波形与接收可靠度分别起什么作用","",
            "两份已训练权重分别交叉测试开关，各臂用同一发射功率、输入图像、路径及快时间噪声。下表均为开发集 256 张、一次信道实现、10 dB 的 PSNR 差（dB）。","",
            "| 冻结权重来自 | 归一化多普勒 | 逐位置可靠度下，开扩展 − 关扩展 | 均匀可靠度下，开扩展 − 关扩展 | 开扩展时，均匀 − 逐位置可靠度 |",
            "|---|---:|---:|---:|---:|"]
    wa=rows(d5/"waveform_val256/paired.csv")
    for tag,title in (("symp_otfs","扩展训练"),("symp_ofdm_2d","OFDM 训练")):
        for kd in (.1,.18):
            key=dict(weight=tag,kd=kd,metric="psnr_db")
            vals=[fmt(one(wa,**key,contrast=c)) for c in ("spread_at_local","spread_at_uniform","uniform_at_spread")]
            lines.append("| "+" | ".join([title,f"{kd:.2f}",*vals])+" |")
    lines+=["","这些是同权重切换时的条件效应。500 图完整系统的 +0.395 dB 使用各自训练的权重。两组证据关联在一起，说明图像表示与发送波形共同训练的重要性。[全部配对](dir5/waveform_val256/paired.csv) · [源图像得分](dir5/waveform_val256/source_scores.csv) · [快时间物理核验](dir5/channel_check.json)",
            "","## 5B · 正式测试与同接收观测的机制干预","",
            "以下比较为 GenHolo 与使用同一冻结 VQ-VAE 的 BlackBox 之差。两次正式评测的模型各按自己的权重哈希归档。PSNR 列为从来源均值计算的差，原文件对 DISTS 和 LPIPS 提供直接配对区间。","",
            "| 版本 | 图像 | 信道带宽比 | 10 dB ΔPSNR (dB) | ΔDISTS | ΔLPIPS |",
            "|---|---|---|---:|---:|---:|"]
    for ver in ("observed_v1","published_v2"):
        data=rows(d5/ver/"all_paired_long.csv")
        for split,name in (("common500","500 张"),("kodak24","Kodak24")):
            for den in (128,256):
                key=dict(split=split,condition=f"cbr{den}_snr10dB",reference="BlackBox_VAR_Blind")
                vals=[fmt(one(data,**key,metric=k),3 if k=="psnr_db" else 4,"paired_mean") for k in ("psnr_db","dists","lpips_alex")]
                lines.append("| "+" | ".join(["v1" if ver=="observed_v1" else "v2",name,"1/"+str(den),*vals])+" |")
    data=rows(d5/"published_v2/all_paired_long.csv")
    lines+=["","与采用真实信道信息的 Swin（训练目标不同）相比，v2 在 500 张图、10 dB 的效果为","",
            "| 带宽比 | ΔPSNR (dB) | ΔDISTS | ΔLPIPS |","|---|---:|---:|---:|"]
    for den in (128,256):
        key=dict(split="common500",condition=f"cbr{den}_snr10dB",reference="Swin_Genie_CSI")
        vals=[fmt(one(data,**key,metric=k),3 if k=="psnr_db" else 4,"paired_mean") for k in ("psnr_db","dists","lpips_alex")]
        lines.append("| "+" | ".join(["1/"+str(den),*vals])+" |")
    lines+=["","固定 v2 的 1/256 接收器，每张验证图只产生一次有噪观测。改变码本修正时沿用同一个后处理网络。连续潜变量还可直接送给冻结解码器，作为第二个干预。","",
            "| 256 张验证图，10 dB | ΔPSNR (dB) | ΔDISTS | ΔLPIPS |","|---|---:|---:|---:|"]
    re=rows(d5/"receiver_val256/paired.csv")
    for a,b,title in (("native","gates_off_same_polisher","原接收器 − 关码本修正（同后处理）"),
            ("gates_off_same_polisher","continuous_only","关码本修正且保留后处理 − 连续潜变量直解"),
            ("native","continuous_only","原接收器 − 连续潜变量直解")):
        vals=[fmt(one(re,left=a,right=b,metric=k),4) for k in ("psnr_db","dists","lpips_alex")]
        lines.append("| "+" | ".join([title,*vals])+" |")
    lines+=["","在这个固定接收器和工作点上，启用码本修正使 DISTS 和 LPIPS 都升高。源图感知指标最低的一臂直接解码连续潜变量。联合训练中的约束可能影响前级表示，上述推理干预无法估计移除该模块后从头训练的结果。","",
            "[5B v1 全表](dir5/observed_v1/) · [5B v2 全表](dir5/published_v2/) · [三臂全部配对](dir5/receiver_val256/paired.csv) · [输入和原验证分数复核](dir5/receiver_val256/identity.json)",
            "","## 资源与复核",""]
    ledger=json.loads((root/"resource_ledger.json").read_text())
    lines.append(f"本轮四臂训练、两项两数据集测试及 5A/5B 两项冻结干预合计 **{ledger['all_new_gpu_device_hours']:.3f} A800 设备小时**，按每个 GPU 工作进程的实际墙钟时间相加。既有生产任务单列为复用。账本逐项给出运行时长与真实退出码。")
    lines+=["","[资源账本](resource_ledger.json) · [逐帧重算和配对区间复核](data_checks.json) · [原六分支输入身份](historical_inputs.json) · [文件链接检查](storage_audit.json)"]
    save_once(root/"CONTROLS_zh.md",("\n".join(lines)+"\n").encode())


if __name__=="__main__":main()
