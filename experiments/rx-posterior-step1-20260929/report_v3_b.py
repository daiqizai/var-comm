"""Render a factual report from the single v3 B publication object."""
import json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'results/rx_posterior_step1_20260929/revision_v3_B'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    decision=read(D/'decision.json');cfg=read(D/'B_frozen_config.json');done=read(D/'B_completion.json')
    summary=list(csv.DictReader((D/'summary.csv').open()));pairs=list(csv.DictReader((D/'paired.csv').open()))
    means={(x['level'],x['mode'],x['profile'],x['metric']):float(x['mean']) for x in summary}
    yes=lambda x:'通过' if x else '未达到'
    lines=['# RX posterior step1 v3：完整 A/B 探针结论','',
       '任务 A、精确极限检查、校准冻结和任务 B 已实际完成。原训练、历史等待器和 var-comm 自动推进继续停止；本次没有训练、没有访问新 holdout，也没有自动启动第二步。','',
       '## 结论','',
       f"- C（Oracle天花板）：通过，见[任务A报告](rx_posterior_step1_v3_task_A_20260929.md)。",
       '- 精确极限检查：通过。η=0、β=0、λ=1；四张固定校准源的TF和CL中，V/A1逐token完全一致且Fq逐位一致。',
       f"- M1准确率判据：{yes(decision['M1_accuracy_passed'])}；保留原对数概率附加条件的联合判据：{yes(decision['M1_original_joint_passed'])}。",
       f"- M2（路径条件准确率与latent平方误差）：{yes(decision['M2_passed'])}。",
       f"- M3（闭环融合图像）：{yes(decision['M3_passed'])}。",'']
    if decision['M3_passed']:
        lines.append('V-fuse在目标1 dB和−2 dB同时优于A1-fuse/A2-fuse，达到M3；4 dB虽有改善，但对A2未达到登记幅度门槛。M1未达到的关键是模糊带内相对A2的增益不足，M2仅10 dB一档通过，低SNR融合前latent误差没有同步降低。因此结论是融合后的感知质量有收益，不能宣称token机制全链成立或通过全部门槛；它也不是N4084实际PHY成绩。')
    else:
        lines.append('本轮没有达到预定图像门槛，不能宣布VAR后验方案有效。C通过只说明真实token先验具有潜力，不能替代可实现接收端的M3。')
    lines+=['','**M3是实际图像价值检验，token指标只说明机制。** 低SNR细尺度难判，不自动否定粗/中尺度消歧，但也不能据此免除图像门槛。所有负结果和区间跨0项均保留。','',
       'M1原条目的对数概率条件没有得到明确取消，本报告按事先声明同时列出两个判据，不按开发集结果择优命名通过。','',
       '## 逐档判定与冻结参数','',
       '| 等效SNR | 模糊带尺度 | λ A2 / V | M1准确率 | M1联合 | M2 | M3 |',
       '|---:|---|---|---|---|---|---|']
    for lev in cfg['levels']:
        if lev['role']!='decision':continue
        d=next(x for x in decision['levels'] if x['level']==lev['name'])
        lines.append(f"| {lev['snr_db']:g} | {','.join(map(str,lev['ambiguity_scales_1based'])) or '无'} | {lev['decision']['A2']['lambda']:g} / {lev['decision']['V']['lambda']:g} | {yes(d['M1_accuracy_pass'])} | {yes(d['M1_original_joint_pass'])} | {yes(d['M2_pass'])} | {yes(d['M3_pass'])} |")
    lines+=['','M1模糊带完全由200张校准源×三噪声seed确定，并在B-development前冻结。每档按模糊带尺度等权准确率选λ，空带用λ=1且M1不适用；A2/V同一网格、同一选择规则。判定β固定为0，β>0仅供概率与可靠性诊断。30 dB只作TF相对诊断，不计入以上七档。','',
        '## 目标低信噪比图像结果','',
        '| SNR | 方法 | PSNR dB | LPIPS | DINO |','|---:|---|---:|---:|---:|']
    baseline=list(csv.DictReader((D/'low_SNR_A_baseline_means.csv').open()))
    for lev in cfg['levels']:
        if not lev['name'].startswith('target_'):continue
        name=lev['name']
        for method in ['B1','O1','A1_decision','A2_decision','V_decision']:
            if method in ['B1','O1']:v={x['metric']:float(x['mean']) for x in baseline if x['level']==name and x['method']==method}
            else:v={m:means[(name,'CL',method,'fused_image_'+m)] for m in ['psnr_db','lpips_alex','dino_cosine']}
            lines.append(f"| {lev['snr_db']:g} | {method.replace('_decision','-fuse')} | {v['psnr_db']:.5f} | {v['lpips_alex']:.6f} | {v['dino_cosine']:.6f} |")
    lines+=['','### 主要配对差与95%区间','',
        '下表为V-fuse减对照；LPIPS越低越好，DINO越高越好。源图先平均三噪声，再配对bootstrap。','',
        '| SNR | 对照 | ΔLPIPS [95% CI] | ΔDINO [95% CI] | ΔPSNR dB |',
        '|---:|---|---|---|---:|']
    for d in decision['levels']:
        if d['snr_db'] not in [4,1,-2]:continue
        for control,c in d['comparisons'].items():
            lp=c['fused_image_lpips_alex'];di=c['fused_image_dino_cosine'];ps=c['fused_image_psnr_db']
            lines.append(f"| {d['snr_db']:g} | {control.replace('_decision','-fuse')} | {lp['mean']:+.5f} [{lp['lo']:+.5f}, {lp['hi']:+.5f}] | {di['mean']:+.5f} [{di['lo']:+.5f}, {di['hi']:+.5f}] | {ps['mean']:+.5f} |")
    lines+=['','1 dB的原P4084近似参照为22.13583 dB、LPIPS 0.131857、DINO 0.888098；本轮V-fuse为19.85412 dB、0.185691、0.818914，仍落后。不同资源/前端只允许近似对照，不能把M3通过表述为胜过P4084。','',
        '1 dB的M2融合前latent平方误差，V相对A1增加98.77（95% CI 29.11至166.10），相对A2增加72.33（9.85至132.74）；两项均显著变差。该指标必须保留，不能用融合后的latent改善替代。','',
        '闭环十尺度等权准确率虽在低SNR提高，但按token数加权的准确率在4/1/−2 dB均下降，说明细尺度并未整体改善；两种汇总均完整报告。']
    lines+=['','B1/O1取已完成任务A中完全相同源图和噪声draw，新增配对差只作旁证，不改变M1–M3。O1依赖真实Fq，不是可部署接收端。','',
      '![M3融合质量](../results/rx_posterior_step1_20260929/revision_v3_B/M3_fused_quality.svg)','',
      '## 30 dB相对诊断','',
      '| 尺度 | A1 | A2固定 | V固定 | A2校准λ | V校准λ |','|---:|---:|---:|---:|---:|---:|']
    for k in range(1,11):
        vals=[means[('relative_30dB','TF',p,f'acc_k{k}')] for p in ['A1_decision','A2_fixed','V_fixed','A2_decision','V_decision']]
        lines.append('| '+str(k)+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines+=['','### 细尺度是否接近随机：用实测核对','',
        '| SNR | A1 第8–10尺度平均 | A2 第8–10尺度平均 | V 第8–10尺度平均 |',
        '|---:|---:|---:|---:|']
    for lev in cfg['levels']:
        if not lev['name'].startswith('target_'):continue
        vals=[sum(means[(lev['name'],'TF',p,f'acc_k{k}')] for k in [8,9,10])/3 for p in ['A1_decision','A2_decision','V_decision']]
        lines.append('| '+str(lev['snr_db'])+' | '+' | '.join(f'{v:.4%}' for v in vals)+' |')
    lines+=['','4096码字均匀随机猜测参考为1/4096≈0.0244%。低绝对准确率不能自动称为“等同随机”；同时必须与真实A1和A2对照。上表只作机制说明，不改变模糊带或M3判定。']
    lines+=['','此处不设99%或其他绝对门槛，也不能用CL相对于原编码路径的一致率代替路径条件判决质量。','',
       '## 闭环与统计','',
       '每个尺度先固定接收端自己的历史判决，用干净F在评分支路确定“当前路径下最近码字”，再评价实际RX选择；干净F不进入RX先验、MAP或融合。真实权重检查中将评分用F改变后，RX tokens和Fq保持逐位相同。latent误差使用原坐标总平方和 ||F−Fq_post||²，未偷换成均方或标准化误差。','',
       'M2路径准确率主统计为十尺度等权平均，另报token数加权值。所有source×SNR×noise×profile对象保留。先对每源三噪声平均，再作10000次源图配对bootstrap（seed20260929，逐点95%百分位区间）。区间不表示训练seed变异，也未作多重比较校正。','',
       '![M2路径和latent](../results/rx_posterior_step1_20260929/revision_v3_B/M2_path_and_latent.svg)','',
       'β>0概率诊断使用同一β=0决策前缀，并不生成另一套可参与M1–M3的MAP输出。CL概率可靠性仍对原编码token评分，因此与M2路径条件目标不同；图中明确区分。','',
       '![V可靠性](../results/rx_posterior_step1_20260929/revision_v3_B/V_reliability.svg)','',
       '## 执行、身份与边界','',
       '- 原100 development源、三seed、七判定档和额外30 dB TF诊断；A1、A2/V判定版以及A2/V固定参照，共22500条完整TF/CL记录。全200张校准网格、200张融合方差统计、100张development原记录以逐源稀疏bin明文JSON封存，可逐字节复原原JSON并校验原SHA。',
       '- 同一源/噪声在先验、前缀模式和SNR档位间配对复用；λ=1时固定版与判定版复用相同推断。22500是完整评测记录数，不是22500个独立噪声样本或实际PHY传输。统计单位仍为100张源图。',
       '- 模型、源码、预处理、原source/seed、校准freeze和每个结果对象的SHA见索引。所有重建统一冻结Dc；Source A复现23.464584 dB、LPIPS .09887115。',
       '- 实际SNR约定为每实数维噪声方差1/γ，η=10^(−SNR_dB/20)。标准化只保证校准分布集合平均能量，非逐帧强制能量。N4096等效latent探针对N4084实际PHY只能近似对照。',
       '- 实测development平均标准化信号功率为0.960869（相对参考约−0.173356 dB）；逐帧功率约0.190915至2.093766。因此同时报告实际能量比，保留预先固定的名义噪声档位，不使用development能量重新调噪声。详见[能量账本](../results/rx_posterior_step1_20260929/revision_v3_controls/signal_noise_energy.json)。',
       '- 原v2 30 dB硬门槛失败保留为历史，并由用户明确替换。v3 CPU端点测试暴露浮点归约问题后，先安全停止校准、归档216份封存数据，修正仅1e−12端点比较容差，回归通过后重算；实际选出的档位/参数未改变。不是GPU质量失败。',
       '- 任务A先读其原development是用户明确要求；A结果没有进入B参数选择，B自身读取development前全部参数和融合方差已冻结。',
       f"- B冻结config SHA：{sha(D/'B_frozen_config.json')}。",
       f"- B完成回执SHA：{sha(D/'B_completion.json')}。",
       '',
       '[全部数据、配对区间、校准与图表索引](../results/rx_posterior_step1_20260929/revision_v3_B/index.json)。根工程与独立远端检查以实际receipt为准，报告文件本身不替代验收。']
    out=ROOT/'reports/rx_posterior_step1_v3_complete_20260929.md'
    assert not out.exists();out.write_text('\n'.join(lines)+'\n');print(out)
if __name__=='__main__':main()
