"""Write delivery notes only after actual result and document acceptance."""
from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parents[1]
review=json.loads((P/'validation/artifact_review.json').read_text())
assert review['status']=='PASS' and review['all_pages_actually_viewed']
for item in review['documents']:
 assert hashlib.sha256((P/item['path']).read_bytes()).hexdigest()==item['sha256']
audit=json.loads((P/'validation/claim_audit.json').read_text())
assert audit['closed_evidence']==['E1','E2','E3','E4','E5','E6']
with (P/'inputs/e5/timing_summary.csv').open(newline='')as f:t=list(csv.DictReader(f))
def mean(g,d):return float(next(v for v in t if v['group']==g and v['direction']==d)['equal_source_mean_seconds'])*1000
pages={v['path']:v['pages']for v in review['documents']}
resources=json.loads((P/'inputs/e5/resource_ledger.json').read_text())
report=f'''# V7 完整执行结果与交付

主配置为 `original_receiver_plus_D_token`：保留原冻结模型、适配后 D_c 与原 argmax 接收机，仅使用逐 token 实际码长装入。本轮没有新训练，也没有恢复此前停止的支线。

## 结果

- E1：66,000 逻辑位置、五指标，220 均值及 220 直接来源配对区间。七个预算/SNR 点的 NeST−WHOLE 五指标区间均有利于 NeST。N1024/10 dB 源码容量利用率为 77.70%→99.79%。完整系统比较仍保留 Ada 的感知优势和区间跨零结果，不宣称全面领先。
- E2：六个总信息预算、15,000 独立接收逻辑结果，90 均值和 90 配对区间。NeST−WHOLE 的 30 个区间均有利。NeST−Ada DISTS 在 620–1940 bits 有利于 NeST，在 2248/2577 bits 有利于 Ada；DreamSim 在 1940 bits 区间跨零，在高两点有利于 Ada。信息预算消除了无线/MCS因素，没有消除 tokenizer、prior、decoder 等系统差异。
- E3：100 源、共同模型/MCS/ceiling/噪声，Whole/Quarter/Eighth/Token 利用率为 78.13/95.19/97.44/99.85%。Token−Quarter PSNR +0.172 [0.125,0.221] dB，DISTS 与 DreamSim 配对区间也有利。实际端点数和检查数分别记录。
- E4：1,800 个实际前缀干预位置。10 dB B0−A0 与 C0−B0 的 PSNR 约 +0.287/+0.537 dB；最后尺度没有更细传播。不是将 Token−Quarter 增益拆成两项。
- E5：16 组、32 个 TX/RX 阶段全部实际退出0，5120 条计时记录（2048 预热、3072 测量），48 个汇总行。N1024/10 dB NeST TX/RX {mean('Token_NeST_N1024_S10','TX'):.1f}/{mean('Token_NeST_N1024_S10','RX'):.1f} ms，WHOLE {mean('WHOLE_N1024_S10','TX'):.1f}/{mean('WHOLE_N1024_S10','RX'):.1f} ms，Ada {mean('Ada_N1024_S10','TX'):.1f}/{mean('Ada_N1024_S10','RX'):.1f} ms。32 源、2 核、同 GPU、同原数值路由，含实际编解码和逐帧 ARPC IPC；BPG CPU-only。全部 median/p95、成功条件 RX 与资源记录在 S7/CSV 中。
- E6：30,000 实际帧，17,497 接收成功、12,503 拒收。8 dB 三法全部拒收，9 dB 误帧率为 WHOLE15.75%、NeST14.95%、Ada13.50%，10 dB 零观察拒收。不是 SNR 鲁棒性保证。

## 计算与复用

E2 新接收重建：NeST/WHOLE 195、Ada 8733；精确接收输入复用分别4466、225。新增评分8882原生图像对，6118逻辑位置复用已核身份的结果。E3为891唯一物理输入中的591新事件+300复用，100新RX+197复用。E4新增394 A0/B0重建，另2个资格回放核对原C0。E6不新增生成，只对88个灰图/源图对补两种距离，其余身份复用。

E5已有14组4480条成功计时原样保留；经用户明确续开最多两小时，仅补ARPC/BPG两组640条完整记录；另保留一次未完成的RX预热及其资源费用。所有旧失败、修订、资格、actual_wait 和原deadline留存。错误的`statistics`导入及BPG子进程核数适配分别修复；模型和源码数学不变。计时代码实际扫描完整候选列表，旧`length_checks`字段是所选端点降序位置；新计数表明确区分实际检查数量和该位置。没有重跑成功主实验。

本包受控作业登记GPU reservation合计 {resources['registered_GPU_reservation_seconds']/3600:.6f} h；包含独立PHY、BPG及历史失败保守余量的CPU合计 {resources['conservative_total_CPU_process_wall_seconds']/3600:.6f} h，均在原72 h上限内。BPG虽仅用CPU，原调度为独占保留了GPU预约，账本保留该保守费用并另计CPU。文件构建、仓库检查另有回执；不冒称模型推理资源。

## 稿件与图表

正文保留同模型收益、匹配粒度、接收端有序干预、低预算表现和完整系统感知取舍。Table II 补齐真实计时；S4/S5/S7给完整信息预算、门限和成本。Fig.1保留可编辑PPTX；Fig.2为N1024六方法4/7/10/19 dB的DISTS/DreamSim及48个逐点CI；Fig.3为7 dB两张按优势选择的全视野案例，明确不称随机或代表样本。旧图与同源全方法补图保留。

实际页数：双栏{pages['manuscript_doublecolumn.pdf']}页、双倍行距单栏{pages['manuscript_singlecolumn.pdf']}页、Word导出{pages['manuscript_review.docx']}页、补充材料{pages['supplementary.pdf']}页。所有最终页面实际打开检查，哈希见`validation/artifact_review.json`。Word正文、三张表和OMML公式均可编辑；SVG字体可编辑性按来源记录说明，Fig.1文字编辑源为PPTX。

## 交付与 Git

入口：`main.tex`、`main_singlecolumn.tex`、`manuscript_doublecolumn.pdf`、`manuscript_singlecolumn.pdf`、`manuscript_review.docx`、`supplementary.pdf`。数据和代码在本稿`inputs/`、`scripts/`及`results/nestcom_perceptual_followup_20261030_v2/`、`experiments/nestcom-perceptual-followup-20261030/`。大数组、模型、日志仍只在PFS。主张、来源和验收对应`claim_evidence_matrix.md`、`figure_table_sources.json`、`validation/`。

全部科学结果与文档验收已完成；正常远端提交/推送及独立remote SHA以最终publication receipt为准，本文件不预填未来提交SHA。本机不commit，不强推。没有需要另开科学实验才能完成的默认项目；新训练、新200或参数扫描均not-run-optional。
'''
(P/'final_report_zh.md').write_text(report,encoding='utf8',newline='\n')
readme='''# NeST-Com V7 — complete artifacts

Main configuration: `original_receiver_plus_D_token`. Title: **NeST-Com: Fixed-Budget Next-Scale Token Communication**. Anonymous review version. Historical V6 and all original scientific records are preserved.

E1–E6 and figure selection are closed. See `final_report_zh.md` for findings, unfavorable results, actual computation/reuse, limitations and delivery paths. `validation/artifact_review.json` binds every actually viewed final page to the delivered document hashes; earlier draft reviews certify only their original files.

The complete entry points are `manuscript_doublecolumn.pdf`, `manuscript_singlecolumn.pdf`, `manuscript_review.docx`, and `supplementary.pdf`. Native LaTeX, PPTX, Matplotlib scripts, editable Word tables/equations, PDF/SVG/PNG figures and full-precision CSVs accompany them. Model weights, native arrays, caches and large logs stay on PFS.

## Rebuild documents only

Use the frozen delivered input CSVs. Run `scripts/build_evidence_tables.py`, `scripts/audit_closed_claims.py`, and `scripts/build_supplement.py`. Compile `main.tex`, `main_singlecolumn.tex` and `supplementary.tex` twice with pdfLaTeX. Set `PANDOC`, `PDFTOPPM` and optionally `NEST_DOC_WORK`, then run `scripts/build_word.py`. It reads the same TeX/CSVs and preserves native text, three tables and OMML equations. Run `scripts/build_source_index.py` after changes and repeat real full-page rendering/review; do not reuse old visual PASS.

Figure sources are in `scripts/` and native Fig.1 in `figures/figure1.pptx`. The fixed scientific figures need no model calls to redraw. Source/data conditions and hashes are in `figure_table_sources.json`; claim populations and limitations are in `claim_evidence_matrix.md`.

Experiment entry points are separately registered. Do not run launchers as part of manuscript compilation, and do not repeat completed timing or statistical stages. Final normal remote-only Git publication is verified in the external publication receipt and final user delivery; no local commit or force push.
'''
(P/'README.md').write_text(readme,encoding='utf8',newline='\n')
claims=(P/'claim_evidence_matrix.md').read_text(encoding='utf8')
claims=claims.replace('# V7 主张与证据（进行中）','# V7 主张与证据')
claims=claims.replace('本表不是 E2/E5/E6 的完成回执。','E2/E5/E6 的独立 actual-wait 完成身份见 inputs 中各自 execution_closure.json。')
claims=claims.replace('done；实际耗时待 E5','done；实际耗时见 E5/S7，早停比较与计时全扫描分列')
lines=claims.splitlines()
for i,line in enumerate(lines):
 if line.startswith('| 相同信息位预算下三系统重建关系'):
  lines[i]='| 相同信息位预算下三系统重建关系 | common500、六 B 值，WHOLE/Token 各一次，Ada 三固定生成重复 | `inputs/e2/summary.csv`、`paired_deltas.csv`、`execution_closure.json`，15000结果 | NeST−WHOLE全30区间有利；NeST−Ada高两预算DISTS/DreamSim不利，1940 bits DreamSim跨零；不隔离模型混杂 | done |'
 if line.startswith('| 端到端处理成本与粒度代价'):
  lines[i]='| 端到端处理成本与粒度代价 | timing32，16组，同PRO6000、2CPU核，2热身+3测量 | `inputs/e5/timing_summary.csv`、`execution_closure.json`；5120记录、48汇总 | 全部实际编解码；BPG CPU-only，ARPC IPC计入；是软件处理成本，不是空口时延；完整候选扫描与早停位置分开 | done |'
 if line.startswith('| 固定 10 dB 策略接近门限的拒收风险'):
  lines[i]='| 固定 10 dB 策略接近门限的拒收风险 | control100、8/8.5/9/9.5/10 dB、20噪声，30k帧 | `inputs/e6/summary.csv`、`execution_closure.json` | 全部失败保留；8 dB全拒收、10 dB零观察失败；不声称低误帧率保证或SNR鲁棒 | done |'
(P/'claim_evidence_matrix.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
outline=(P/'outline_zh.md').read_text(encoding='utf8').replace('III.E（待闭合）','III.E（已闭合）')
start=outline.index('主文优先三图三表')
end=outline.index('\n\n补充图确定',start)
outline=outline[:start]+f"主文采用三图三表、双栏10pt，实际{pages['manuscript_doublecolumn.pdf']}页；单栏{pages['manuscript_singlecolumn.pdf']}页、Word{pages['manuscript_review.docx']}页、补充{pages['supplementary.pdf']}页。全部最终页面已实际打开，哈希绑定记录为 validation/artifact_review.json。"+outline[end:]
(P/'outline_zh.md').write_text(outline,encoding='utf8')
(P/'CHANGELOG.md').write_text('''# V7 change log

Historical V6, original protocols, raw failures, original receiver/model mathematics and closed scientific results are preserved.

- Main configuration is original_receiver_plus_D_token, with 651 legal token endpoints and actual independently terminated lengths; original Quarter remains a labelled control.
- E1 adds native DISTS across 66,000 logical outcomes and retains all other metrics, exact failures and matching-route 7-dB CLIP completion.
- E2 completes six information budgets, exact independent source decoding, all five metrics and mixed system outcomes.
- E3 matches model/MCS/ceiling/noise; E4 uses actual-prefix ordered interventions. Neither becomes a decomposition of unrelated population gains.
- E6 retains all 30,000 real receptions, gray failures and conditional quality; the text does not claim SNR robustness.
- E5 completes all 16 groups and actual TX/RX spans. The final explicitly authorized two-hour extension only adds ARPC/BPG, preserving 14 completed groups. Repairs retain model/precision/math and original failures. Full-scan timing counts are distinguished from selected endpoint ranks and quality-control early stopping.
- Fig.1 preserves native PPT geometry, with updated text. PPTX is its editable text source; SVG fonts are paths. Fig.2 contains all 48 six-system DISTS/DreamSim means and intervals at numeric SNR positions. Fig.3 uses the disclosed two selected 7-dB advantages, with full-field original native outputs and same-source full-method alternatives.
- All three tables, captions, LaTeX and native Word tables share the delivered CSVs. Word body and OMML equations remain editable; actual counts are recorded by the conversion validator.
- Main, double-spaced single-column, Word and supplement are rebuilt after complete E2/E5/E6 delivery. Every final page is actually inspected; old draft reviews are retained only as history.
- Required repository checks and normal remote-only Git publication use fresh receipts. No local commit, forced push, training or stopped research restart.
''',encoding='utf8')
checks=[('Frozen package / sources / policies','done','Exact protocol and source hashes'),('No training or stopped branch restart','done','Only authorized inference/PHY/scoring/timing'),('E1 five-metric comparison','done','66000 logical;220 means/220 pairs'),('E2 information budgets','done','15000 logical;90 means/90 pairs'),('E3 matched granularity','done','1200 logical;20 means/30 pairs'),('E4 receiver interventions','done','1800 logical;30 means/30 pairs'),('E5 complete timing','done','16 groups;5120 records;actualwait0/gone'),('E6 fixed-policy scan','done','30000 frames;15 summary cells'),('Figure1/2/3 and same-source supplement','done','Native sources and hash-bound review'),('PDF / single-column / Word / supplement','done','All final pages actually viewed'),('Claims / five metrics / unfavorable results','done','claim_audit.json and complete CSVs'),('Compute / reuse / failures','done','resource_ledger.json and preserved_failures.json'),('Repository checks and publication','pending-final-publication-receipt','Normal remote-only checks/commit/push; exact result in final receipt'),('New training / new200 / parameter scan','not-run-optional','Outside this execution package')]
(P/'completion_checklist.md').write_text('# Completion checklist\n\n| Item | State | Evidence |\n|---|---|---|\n'+'\n'.join('|'+ '|'.join(v)+'|'for v in checks)+'\n',encoding='utf8')
print(json.dumps(dict(status='DELIVERY_NOTES_WRITTEN',pages=pages)))
