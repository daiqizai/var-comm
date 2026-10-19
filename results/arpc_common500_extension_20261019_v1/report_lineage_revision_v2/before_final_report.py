"""Report saved results and preserved engineering lineage; no model or statistics."""
import collections,time
from common import *
LABELS={'psnr_db':'PSNR (dB)','lpips_alex':'LPIPS-Alex','dists':'DISTS','dreamsim':'DreamSim','dinov2_vitl14_cosine':'DINO-L cosine','convnext_top1_source_prediction':'Prediction agreement (fraction)','ms_ssim':'MS-SSIM'}

def main():
    report=O/'final_report_v1';assert not report.exists()
    done=read(O/'delivery_v1/completion.json');assert done['sources']==500 and done['logical_frames']==4500 and done['unresolved']==0
    for d in done['outputs']:verify(d)
    for stage in ('source','link','rx','metrics'):
        r=read(O/(stage+'_v1')/'root_closure_review.json');assert r['status']=='PASS'and r['all_processes_gone']and r['unresolved']==0
        for d in r['evidence']:verify(d)
    visual=read(O/'figures_v1/actual_visual_review.json');assert visual['status']=='PASS'and visual['actual_opened_color_and_grayscale']
    for d in visual['files']:verify(d)
    for role in ('statistics','figures'):
        control=L/(role+'_control_v1');check_wait(control/'child.actual_wait.json');assert gone(read(control/'supervisor.json')['process'])
    recovery=read(O/'link_finalization_v1/recovery_audit.json')
    assert recovery['status']=='PASS_WITH_DISCLOSED_ORIGINAL_ATTEMPT_LOG_LOSS'
    comp=read(O/'delivery_v1/actual_compute_and_reuse.json');paired=csvrows(O/'delivery_v1/nest_minus_arpc_paired.csv')
    comparison=csvrows(O/'delivery_v1/original100_vs_common500.csv');changed=[x for x in comparison if x['conclusion_changed']=='True']
    baseline=read(O/'baseline_inputs_v1/completion.json');assert baseline['material_mean_differences']==0
    drift=read(O/'delivery_v1/old100_baseline_version_check.json')
    report.mkdir()
    evidence=[]
    scopes=['source_prepare_control_v1','source_dispatch_control_v1','link_prepare_control_v1','link_dispatch_control_v1','link_finalization_control_v1','rx_prepare_control_v1','rx_prepare_control_v2','rx_prepare_control_v3','rx_dispatch_control_v1','metrics_prepare_control_v1','metrics_dispatch_control_v1','statistics_control_v1','figures_control_v1']
    for name in scopes:
        root=L/name
        if not root.exists():continue
        for p in sorted(root.iterdir()):
            if p.is_file()and not p.is_symlink()and (p.suffix=='.json'or p.name.endswith(('.stdout','.stderr'))):
                pin=desc(p);copy=None
                if pin['bytes']<=1000000:
                    target=report/'execution_receipts'/name/p.name;target.parent.mkdir(parents=True,exist_ok=True)
                    raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['sha256']
                    with target.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
                    copy=desc(target);assert copy['sha256']==pin['sha256']
                evidence.append({'source':pin,'copy':copy,'large_original_log_retained_at_source':copy is None})
    put(report/'closed_receipt_copies.json',evidence)
    failure={'original_phy_rx_returncode':1,'original_owner_returncode':1,'saved_events':3600,'header_decodes_complete':3600,'body_decodes_complete':3600,'new_scientific_calls_for_recovery':0,'original_denied_open_paths':'UNKNOWN: list existed only in memory and was not saved','receiver_boundary':'The bound original audit hook raised PermissionError before each disallowed open; the hook was not disabled. Saved actual observations and accepted payload bytes were audited after exit.','attempt_free_gate':False,'classification':'Saved-output completion with disclosed original attempt-telemetry loss; not an original exit-0 execution','audit':desc(O/'link_finalization_v1/recovery_audit.json'),'closure':desc(O/'link_v1/root_closure_review.json'),'rx_preparation_failures':'v1: historical clean-closure schema mismatch; v2: original source GPU no longer met utilization admission. Both before native RX/model calls. v3 normalizes the precise schema and selects an available admitted device, with unchanged bounds and native math.','native_host_admission':desc(O/'host3781_admission_v2/admission.json')}
    put(report/'engineering_lineage.json',failure)
    table=['| SNR | Metric | NeST − ARPC | Source-paired 95% CI |','|---:|---|---:|---|']
    for r in paired:table.append(f"| {r['snr_db']} | {LABELS[r['metric']]} | {float(r['mean']):+.6f} | [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}] |")
    changes=['| SNR | Metric | Original100 | Common500 |','|---:|---|---|---|']
    for r in changed:changes.append(f"| {r['snr_db']} | {LABELS[r['metric']]} | {r['old100_conclusion']} | {r['new500_conclusion']} |")
    text=f'''# ARPC N1024 common500 扩展交付

500 源及 4500 个 ARPC 逻辑事件已完成，4/10/19 dB 各 1500 个事件。原 external_common100 的 100 源、900 事件及成绩经核验复用；新增400源保持原 source_id 和 source_index，不重新编号。失败与灰图保留在全部分母。

## 实际新增计算与复用

- 源端：400 caption、400 tokenizer、400 概率遍历，生成2800个e1–e7实码流；TX clean render为0。
- 链路：{comp['new_logical_events']}逻辑事件，{comp['new_encoded_waveforms']}波形编码，{comp['new_header_decodes']}头部与{comp['new_body_decodes']}正文译码；没有为工程收尾重跑这些调用。
- RX：{comp['new_unique_received_states']}独特实际接收状态，{comp['new_rx_prefix_calls']}新前缀恢复、{comp['new_rx_image_renders']}新图像恢复，{comp['reused_rx_states']}既有状态复用。
- 评分：{comp['new_metric_pairs']}独特新图像对，保留3600逻辑评分帧，逻辑复用{comp['within_new_population_logical_score_reuse']}。精确各模型调用数见actual_compute_and_reuse.json。
- 原四方法42000个逐源指标值直接复用，模型推理与评分新增0。原四方法主图48个均值核对无实质差异（容差1e-10）。
- 已登记10000×500、seed2026100701的索引bank复用；不抽取新随机索引，不使用原100源bank。

## 完整结果

delivery_v1/common500_summary.csv给出五方法×三SNR×七指标的105个均值与逐点95%区间；all_method_source_means.csv保留完整52500个逐源指标单元。endpoint_status_distribution.csv和logical_receivers.json给出实际回退终点及接收状态。

先在同源内平均三次噪声，再对500源等权汇总；NeST−ARPC先形成逐源差，再重采样。这里是“来源配对”，历史P/Swin/BPG噪声与ARPC不同，不能称五方法共用相同噪声实现。区间未经多重比较调整。

{chr(10).join(table)}

## 原100源与完整500源

21个直接配对结果中，{len(changed)}个区间方向结论发生变化（有利/不利/跨零分类），全部原精度数值见original100_vs_common500.csv。原100源使用的四方法成绩与当前修正版逐源值共有{drift['changed_source_cells']}个单元不同；若该数非零，变化同时包含基线版本修复与人口扩展，不能全部归因于样本量。

{chr(10).join(changes)if changed else '所有21个区间方向分类保持一致；数值与区间宽度仍以完整500源为准。'}

## 工程失败披露

原PHY RX在完成3600头部和3600正文译码并保存全部数据后，以退出码1结束。触发最终assert的被拒绝文件访问路径列表只在内存中，未落盘，因此确切路径未知。未将其伪称为空，也未将原运行改称exit0。原隔离钩子对越界open先抛PermissionError；后续仅审核已保存的实际接收数据及字节一致性、补齐收尾元数据，没有新链路或模型调用。该结果保留“拒绝尝试日志缺失”的可追溯性限制，详engineering_lineage.json及原失败现场。

原RX准备的schema与GPU准入两次工程失败亦保留。新主机使用经审查的同原模型/数值环境；Python构建日期差异在主机准入证据中披露，不声称主机二进制完全相同。

## 主图与大纲

figures_v1/figure2_common500.pdf、.svg为真正矢量图，.png为600dpi预览；figure2_data.csv只读取上述已保存结果。彩色与灰度导出已实际打开，审查记录见actual_visual_review.json。英文图注及outline_changes.md同目录。Fig.2更新为五方法common500；旧ARPC100表作为历史结果保留。同先验WHOLE/PARTIAL消融及利用率保留，N2048仍为已完成100源，明确低SNR自适应与19dB完整表示区别。

本轮未扩展N2048、SNR、其他基线或计时，未训练或改写整篇论文。模型、原始数据及大型中间缓存不进入Git；重型备份按用户要求延后。仓库检查、提交及推送结果由独立实际发布收据证明，不在本报告预填。
'''
    (report/'REPORT_zh.md').write_text(text,encoding='utf8')
    put(report/'completion.json',{'status':'SAVED_RESULTS_REPORTED_WITH_DISCLOSED_ENGINEERING_LIMITATION','source_count':500,'logical_frames':4500,'summary_cells':105,'paired_cells':21,'conclusion_changes':len(changed),'uncompleted_scientific_items':[],'new_scientific_calls':0,'inputs':[desc(O/'delivery_v1/completion.json'),desc(O/'figures_v1/actual_visual_review.json'),desc(O/'baseline_inputs_v1/completion.json')],'outputs':[desc(p)for p in sorted(report.rglob('*'))if p.is_file()]})
    print(json.dumps(desc(report/'completion.json')))

if __name__=='__main__':main()
