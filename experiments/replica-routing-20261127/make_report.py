"""Write the Chinese evidence report directly from frozen selections and paired results."""
import csv
import json
import sys
from pathlib import Path

def main(base):
    def read(p):
        with (base/p).open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
    def js(p):
        return json.loads((base/p).read_text())
    def row(data,**keys):
        a=[r for r in data if all(r[k]==v for k,v in keys.items())]
        assert len(a)==1,keys
        return a[0]
    def val(r,d=3):
        return float(r["estimate"])
    def band(r,d=3):
        return f'{float(r["estimate"]):+.{d}f} [{float(r["lo"]):+.{d}f}, {float(r["hi"]):+.{d}f}]'
    def absolute(r,d=2):
        return f'{float(r["estimate"]):.{d}f} [{float(r["lo"]):.{d}f}, {float(r["hi"]):.{d}f}]'
    a=read("analysis/mechanism_paired.csv")
    qp=read("analysis/quality_paired.csv")
    qm=read("analysis/quality_means.csv")
    tm=read("analysis/timing.csv")
    se=read("analysis/secondary.csv")
    pct=read("analysis/quality_recovery_fraction.csv")
    cond=["nearfar_14db","nearfar_18db_ood"]
    lines=["# 四用户过载上行中的干扰抵消与有限接收计算",
           "",
           "同一对接收天线同时观测四个用户各自发送的不同图像。接收机迭代三轮，重建各用户的干扰波形，再分离剩余用户。本次比较沿用三个冻结检查点。先在相同接收波形上检查弱用户的分离过程，再测一次接收最多承担三次 RGB 解码及重编码时的质量与端到端接收时延。",
           "",
           "## 干扰误差在哪一步改变排序",
           "",
           "线性副本直接取解混潜变量。潜变量副本先经过已训练的潜变量去噪器。RGB 副本把去噪后的估计先解成图像，再经共享编码器和共享发端适配器生成潜变量以重调制。不同用户使用各自的 Walsh 签名。机制对照逐元素重用原潜变量接收机的十二个抵消权重，并逐轮验证弱用户的实际 LMMSE 合并矩阵完全一致。",
           "",
           "下表给出第三轮的来源组配对差。每项测量都以左列指定的系统作差。前五行使用误差 dB，负值较好。PSNR 正值较好，DISTS 负值较好。括号为逐项 95% 区间。",
           "",
           "| 观测量 | 潜变量减线性，14 dB | 潜变量减线性，18 dB | RGB 减潜变量，14 dB | RGB 减潜变量，18 dB |",
           "|---|---:|---:|---:|---:|"]
    for metric,name in (("antenna_db","双天线干扰误差"),("combiner_db","经弱用户合并器的干扰误差"),
            ("interference_noise_db","经合并器的干扰加噪声误差"),("latent_db","解混潜变量误差"),
            ("frontend_db","图像解码器输入潜变量误差"),("weak_psnr","弱用户图像 PSNR，dB"),("weak_dists","弱用户图像 DISTS")):
        vs=[band(row(a,round="3",metric=metric,comparison=x,condition=c)) for x in ("latent-linear","rgb-latent") for c in cond]
        lines.append("| "+" | ".join([name]+vs)+" |")
    lines+=["","![干扰副本经过接收合并和重建各环节的配对差](figures/replica_geometry.png)","",
        "线性副本在两根天线上的总干扰误差较低。弱用户合并器对不同空间方向赋予不同权重，经过已实际使用的同一个合并器，潜变量副本在两个近远差下都把误差降得更低。比较只改变副本，能把这个排序反转定位在接收合并这一步。18 dB 下再计入噪声后，该项差异的区间跨零。",
        "",
        "RGB 与潜变量的取舍发生在更后面。14 dB 下，RGB 的合并后干扰误差较高，源潜变量误差和图像失真却改善。18 dB 下，两项潜变量误差差值的区间均跨零，RGB 仍提高 PSNR，同时 DISTS 变差。逐图像素误差与感知失真给出了不同的判断。这些结果不足以把所有收益归因于某一种未测的语义特征。",
        "",
        "测量中的分母随对象而定。天线误差用真实数据子空间中的干扰能量归一化，合并后误差用同一合并后的真实干扰能量。加入噪声的一行以合并后的期望用户信号能量为分母。后两项分别比较发送源潜变量以及在相同收端元数据下通过前端的干净潜变量。真信道、发端潜变量、发端图像只进入离线评分。旧研究中三种副本逐组的 PSNR、DISTS 均完全复现，干扰 NMSE 最大绝对差小于 0.000002 dB。",
        "",
        "## 三次 RGB 调用怎样分配",
        "",
        "一次调用指对一个用户做一次本地 RGB 解码及重编码。每次接收仍要输出四幅最终图像。全 RGB 接收对四个用户各做三次，共十二次。有限策略每帧最多三次。接收机依据估计信道功率识别最弱用户和干扰者，部署时看不到真实功率顺序。每个策略在实际抵消轨迹上计算自己的权重。",
        "",
        "开发集包含 256 张图像，即 64 组，每组在两种近远差下各测一次。十五个候选由三套轮次分配与五条用户选择规则组成。按开发集弱用户平均 PSNR，固定排序选择分配 (0,1,2)，各轮按估计接收功率选最强干扰者。自适应排序选择 (0,0,3)，最后一轮把其余三个用户全部选中。在四用户均被检测为活跃时，末轮全选使该规则的排序分数没有实际选择作用。",
        "",
        "正式测试包含 500 张图像构成的 125 个无交叠四图组，三个冻结检查点和两种近远差。除旧功率顺序的复现外，还实测了新信道和噪声下轮换发射功率所属用户的情形，避免最弱身份固定为第四槽。下表为轮换测试。先在四图组内平均检查点，再对四图组配对。两种布局共享图像，分别报告。",
        "",
        "| 接收策略 | RGB 调用/帧 | 14 dB 弱用户 PSNR | 18 dB 弱用户 PSNR | 14 dB 弱用户 DISTS | 18 dB 弱用户 DISTS | 14 dB 接收时间，ms | 18 dB 接收时间，ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,name in [("latent","仅潜变量"),("best_fixed","固定功率排序 (0,1,2)"),("matched_fixed","末轮全选，固定排序"),
                     ("adaptive","末轮全选，按合并器泄漏排序"),("full","全 RGB")]:
        obs=lambda c,m:row(qm,layout="rotated",condition=c,metric=m,arm=arm)
        vs=[name,f'{val(obs(cond[0],"rgb_calls")):.2f}']+[f'{val(obs(c,m)):.{2 if m=="weak_psnr" else 4}f}' for m in ("weak_psnr","weak_dists") for c in cond]
        vs += [absolute(row(tm,condition=c,quantity="mean_rx_ms",comparison=arm),1) for c in cond]
        lines.append("| "+" | ".join(vs)+" |")
    lines+=["","![同一接收配置的弱用户 PSNR 与整帧时延](figures/quality_cost.png)","",
        "横向区间按计时的 32 个四图组计算，纵向区间按质量评估的 125 个四图组计算。括号内给出每轮本地 RGB 调用次数。纵轴展示轮换功率、重新抽取信道和噪声后的弱用户质量。显著性以逐组直接作差的区间判断。"]
    getq=lambda c,m,op:row(qp,layout="rotated",condition=c,metric=m,comparison=op)
    lines+=["","| 预先指定的配对比较 | 14 dB | 18 dB |","|---|---:|---:|"]
    for label,m,op in [("末轮全选减最强固定策略，弱用户 PSNR，dB","weak_psnr","adaptive-best_fixed"),
        ("末轮全选减同轮次固定策略，弱用户 PSNR，dB","weak_psnr","adaptive-matched_fixed"),
        ("固定三次减仅潜变量，弱用户 PSNR，dB","weak_psnr","best_fixed-latent"),
        ("固定三次减仅潜变量，弱用户 DISTS","weak_dists","best_fixed-latent"),
        ("全 RGB 减固定三次，弱用户 PSNR，dB","weak_psnr","full-best_fixed")]:
        lines.append("| "+" | ".join([label,*[band(getq(c,m,op)) for c in cond]])+" |")
    fr=[row(pct,layout="rotated",condition=c,arm="best_fixed",metric="fraction_of_full_minus_latent_weak_psnr") for c in cond]
    if not (base/"analysis/other_users.csv").exists():
        from other_users import main as compute_other_users
        compute_other_users(base)
    other=read("analysis/other_users.csv")
    group_values=[band(getq(c,"mean_psnr","best_fixed-latent")) for c in cond]
    other_values=[band(row(other,condition=c,comparison="best_fixed-latent")) for c in cond]
    lines+=["",f"固定三次相对仅潜变量的四用户平均 PSNR 差为 {group_values[0]} 和 {group_values[1]} dB。单独计算其余三个用户的等权平均，差值为 {other_values[0]} 和 {other_values[1]} dB。此项按每个四图组的原始分数推得，未参与策略选择。"]
    recovery=["{:.1f}% [{:.1f}%, {:.1f}%]".format(*(100*float(v[k]) for k in ("estimate","lo","hi"))) for v in fr]
    speed=[row(tm,condition=c,quantity="ratio",comparison="best_fixed/full") for c in cond]
    saved=[100*(1-val(v)) for v in speed]
    lines+=["",f'以仅潜变量为起点，固定三次调用实现了十二次 RGB 调用所带来弱用户 PSNR 增益的 {recovery[0]} 和 {recovery[1]}。其整帧平均接收时延比全 RGB 分别低 {saved[0]:.1f}% 和 {saved[1]:.1f}%。末轮同时处理三个干扰用户可在一个批次内完成，(0,1,2) 分配要分两轮处理。两者承担相同的三次图像重编码，实测整帧时延仍有差别。',
           "","## 真正更换干扰用户的独立对照","",
           "开发集冻结结果揭示末轮全选的退化情形后，我们登记了单独的次要对照。登记时正式流程已写出一个条件文件，选择只读取开发集候选表。剔除选项顺序无法改变所选用户集的轮次分配后，开发集得分最高的自适应策略选择 (0,1,2)，按对当前最弱用户合并器的估计泄漏强度排序。对照使用完全相同的轮次分配，按估计接收功率排序。单独评估这一个新策略，固定对照质量复用同源同配置的 750 条正式记录。两者时延在新的同场会话中交错重测。","",
           "| 轮换布局，泄漏排序减功率排序 | 14 dB | 18 dB |","|---|---:|---:|"]
    for label,met,op in [("实际选人不同的碰撞比例","fraction","different_active_user_choices"),
            ("弱用户 PSNR，dB","psnr","adaptive_minus_fixed"),("弱用户 DISTS","dists","adaptive_minus_fixed"),
            ("四用户平均 PSNR，dB","all_psnr","adaptive_minus_fixed"),("接收时延比，泄漏排序 / 功率排序","ratio","adaptive_div_fixed")]:
        values=[row(se,condition=c,metric=met,comparison=op) for c in cond]
        fmt=absolute if met in ("ratio","fraction") else band
        lines.append("| "+" | ".join([label,*[fmt(v,3) for v in values]])+" |")
    va=[row(se,condition=c,metric="psnr",comparison="adaptive_minus_fixed") for c in cond]
    if all(float(v["hi"])<0 for v in va):
        conclusion="在两个功率跨度下，单纯按合并器泄漏大小选人都降低了弱用户平均 PSNR。"
    elif all(float(v["lo"])>0 for v in va):
        conclusion="在两个功率跨度下，按合并器泄漏大小选人都提高了弱用户平均 PSNR。"
    else:
        conclusion="这项比较未在两个功率跨度上同时获得弱用户 PSNR 的正向配对区间。"
    verdict=js("analysis/audit.json")["rule_comparison"]
    lines+=["",conclusion+"合并器输出能解释副本质量的排序反转，并不意味着干扰泄漏大就值得优先做昂贵的图像重建。选择动作还取决于 RGB 相对潜变量能修正多少误差。",
       "","## 计时、证据范围与长文定位","",
       "端到端接收时间从已在 GPU 上的四用户混合波形开始，结束于四个用户的 GPU RGB 输出。单碰撞批量，计时前后同步，含选人、三轮抵消、LMMSE 重解与最终图像解码。每种近远差使用前 32 个四图组、固定同一个检查点，每组两次交错测量。区间以四图组为单位。A800 为共享机器，给出实测范围。开发筛选使用同型号的另一块 A800，质量复核、正式测试、主要及次要计时均使用 GPU 3。",
       "",
       "主比较预设条件同时要求轮换布局下的弱用户 PSNR 配对区间下端高于零，且自适应 / 最强固定的接收时间比上端不超过 1.05。14 dB 与 18 dB 的联合判据依次为 "+ "、".join("达到" if v["paired_primary_endpoint_pass"] else "未达到" for v in verdict)+"。",
       "",
       "长文可将干扰副本的评价目标从天线端 NMSE 推进到弱用户合并后的误差，再展示图像质量与计算开销。每帧三次的简单功率排序是实际可用的中间计算点。文中应同时给出全 RGB、仅潜变量和固定功率排序，以便读者选择。四用户均活跃、两个接收天线、两种功率跨度和这套冻结编码器构成直接证据的范围。稀疏活跃用户检测、真实射频测量与其他调制器需各自的数据。",
       "",
       "Xie 等已在双用户 MIMO 上行中把强用户重建图像重新编码，用于弱用户的串行干扰抵消。本研究在四用户双天线的并行接收机内给出副本误差在哪个环节反转的直接测量，以及接收计算受限时不同选择规则的配对比较。本地 RGB 副本采用本系统的共享模型。 [Xie 等，式 (2)](https://arxiv.org/html/2504.07969v1#S2.SS2)",
       "",
       "完整的逐组质量、各轮误差、时延原始值和区间见 [数值汇总](analysis/)。开发集全部 15 个候选见 [筛选表](development/candidates.csv)。[复现说明](../../experiments/replica-routing-20261127/REPRODUCE.md) 包含源数据和运行入口。"]
    with (base/"REPORT_zh.md").open("x",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")

if __name__=="__main__":
    main(Path(sys.argv[1]))
