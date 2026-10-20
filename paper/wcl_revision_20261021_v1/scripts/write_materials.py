from pathlib import Path
import csv,json,shutil,hashlib
H=Path(__file__).resolve().parent;P=H.parent;W=P.parents[1]
D=P/'sensitivity'
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(n,t):(P/n).write_text(t.strip()+'\n',encoding='utf-8')
S=P/'sensitivity';S.mkdir(exist_ok=True)
for n in ['summary.csv','paired_differences.csv','actual_compute_and_reuse.json','statistics_identity.json','completion.json']:
 if (D/n).resolve()!=(S/n).resolve():shutil.copyfile(D/n,S/n)
captions={
'Fig. 1':"""NeST-Com under a fixed complex-symbol budget. The transmitter selects a within-scale endpoint using the actual terminated arithmetic-code length, including the paid length and CRC fields. The receiver entropy-decodes transmitted tokens, predicts the boundary scale, overwrites received positions, and generates finer scales before frozen-codebook aggregation and image decoding. The 10 dB bars report common500 mean entropy-bitstream capacity utilization, defined as terminated stream bits divided by usable source capacity after the 13-bit length and 16-bit CRC fields; this is the definition used in Table I. The lower example illustrates m = 7 and K = 75, rather than a universal operating endpoint.""",
'Fig. 2':"""Five-method comparison on common500 over AWGN at N = 1024 complex symbols per image. Each source is first averaged over its three original noise realizations; markers show means over 500 sources and error bars show pointwise 95% source-bootstrap confidence intervals. LPIPS uses AlexNet. CLIP-I is OpenAI CLIP ViT-L/14 image–image feature cosine similarity, evaluated in FP32 with the frozen 224-pixel preprocessing. Higher PSNR and CLIP-I and lower LPIPS and DreamSim are favorable. The entropy-coded NeST-Com results include the T1-C m10 correction at 19 dB. Comparisons are paired by source, not by identical cross-method noise realizations.""",
'Fig. 3':"""Previously selected development examples at N = 1024 and 10 dB: Pastries (source index 95) and Shopping cart (75). Columns show the original image and the five frozen methods. Within each row, the outlined region uses identical pixel coordinates and a common 2.667× relative enlargement in every column; the full field of view remains visible. Values below reconstructions are whole-image PSNR (dB) / LPIPS-Alex for the displayed single realization, computed from the saved native float outputs. These selected examples are qualitative illustrations and are not random common500 samples.""",
'Table I':"""Effect of within-scale endpoints at N = 1024 on common500. Utilization is terminated entropy-bitstream length divided by usable source capacity after the length and CRC fields. Differences are PARTIAL − WHOLE with direct source-paired 95% bootstrap intervals. Positive ΔPSNR and negative ΔDreamSim favor PARTIAL.""",
'Table II':"""Larger-budget comparison at N = 2048 on matched100. Differences are NeST − ARPC with direct source-paired 95% bootstrap intervals after averaging three noise realizations per source. The 4 and 10 dB rows use the frozen adaptive settings; 19 dB uses full representations. Each method delivered 300/300 events per condition. These rows are distinct configurations. At 19 dB, the retained DISTS result is ΔDISTS = +0.0054 [ +0.0024, +0.0086 ], favoring ARPC, despite NeST's higher PSNR.""",
'Table III':"""Mean software processing latency at 10 dB, in ms/image, for the frozen N = 1024 implementations on one NVIDIA RTX PRO 6000 Blackwell GPU and the same six allocated CPU cores; batch size is one. Each of 16 fixed development sources has three warmups followed by three measurements, giving 48 measured calls per method. TX and RX are measured separately; Total = TX + RX. Model loading, simulated AWGN generation, process handoff, and audit/result persistence are excluded; BPG codec subprocess and necessary file-conversion costs are included. The original per-method framework and precision settings and sampled shared load are retained. These are software processing times, not radio airtime, and are not merged with the historical H800 measurements."""
}
write('captions.md','# English captions\n\n'+'\n\n'.join('## '+k+'\n\n'+v for k,v in captions.items()))
sections=[
('I. Introduction',[
('段落 1：问题与评价目标','建立逐图固定复符号预算 N、给定信道条件下的图像通信问题。说明像素保真、感知相似性、语义表征一致性和处理时间是需要同时考察的取舍，不把四个相关指标称为四个独立证据。'),
('段落 2：相关工作与有限技术缺口','介绍连续潜空间 JSCC、SwinJSCC 及数字源信道分离。加入 Ada-TokenCom：它已将自回归 token、算术编码与头部 token 传输／尾部生成结合，并进行源率与 MCS 适配。TONIC 可简述为任务导向的 token 保护与接收补全工作。技术缺口限定为 next-scale 多尺度表示的端点粒度与逐图硬预算匹配，不把“发送一部分、生成其余”本身宣称为首创。'),
('段落 3：本工作的连贯设计及证据','将尺度内前缀、实际终止熵码长适配、前缀约束重建作为一套完整方法。WHOLE/PARTIAL 对照验证内部机制；五方法 common500 展示系统表现；N2048 和 ARPC 解码敏感性说明方法边界。避免预设全面优越。')]),
('II.A. System Model and Multiscale Representation',[
('段落任务','定义 x、冻结 tokenizer、多尺度离散 token、共享 VAR 先验与 Dc。N1024 为 68 头部 + 956 正文复符号，AWGN，标称 Es = 2。源图 256×256。载荷包含实际终止、长度、CRC 与对齐；不能把源位数与符号数混为一谈。引用 Fig.1(a)。'),
('表示与接收可用信息','区分完整尺度 1…m 与下一尺度按固定 raster 顺序的 K 个 token。RX 仅有实际收到的付费 profile、合法码流及共享模型；原图、未传 token 和 TX CDF 不可作为接收端输入。')]),
('II.B. Budget-Adaptive Within-Scale Prefix Transmission',[
('段落 1：端点与冻结策略','按实际候选表说明 WHOLE 和 PARTIAL 的合法端点及 SNR 对应冻结策略；不要把图中 m=7、K=75 的解释性示例写成所有图统一配置。calibration 选择与测试人口分离，19 dB 使用已完成 T1-C 的 m10 修正。'),
('段落 2：实际码长适配','将终止算术码长 Lterm 纳入核心约束 Lterm + 13 + 16 ≤ k；头部和正文总符号数固定。按照现有合法端点顺序实际回退，装不下或拒收仍保留分母。不能用估计熵或整段码长差替代实际已终止的流。'),
('机制解释','容量利用率定义为 Lterm/(k−13−16)，不是 token 数比例，也不是每符号频谱效率。10 dB 的 WHOLE 77.70%、PARTIAL 95.21% 与 Fig.1 和 Table I 同源同定义。')]),
('II.C. Prefix-Constrained Reconstruction',[
('段落 1：恢复和生成的顺序','先用共享先验和已解码上下文恢复实际传输 token；在边界尺度预测所有位置，再覆盖真实收到的位置，合并后推进状态并生成更细尺度，最后码本查找、聚合和冻结 Dc 解码。引用 Fig.1(b)。'),
('段落 2：约束与失败','仅收到的 token 可以覆盖预测。保持现有采样方式、生成种子与接收拒收规则；源解码失败输出真实规定结果。明确熵解码恢复已传信息与生成未传表示的区别。不得将干净前缀图当作无线性能证据。')]),
('III.A. Experimental Setup',[
('段落 1：主比较与版本','Fig.2 使用 N1024、4/10/19 dB、原 common500。五方法为 entropy-coded NeST-Com、Latent JSCC 最终100k、SwinJSCC adapted 最终200k、现有 adaptive BPG + LDPC、adapted ARPC + LDPC。NeST 19 dB 用当前 T1-C PARTIAL；不混入旧 raw 或旧检查点。'),
('段落 2：评价与统计','PSNR、LPIPS-Alex、DreamSim、CLIP-I；CLIP-I 是 OpenAI CLIP ViT-L/14 的图像—图像余弦，224 输入、冻结浮点预处理、FP32，不做 softmax 或百分比变换。三次原噪声先按每源平均，500 源等权；点位 CI 沿用既有 source-bootstrap。差异读直接来源配对 CI；历史方法噪声命名空间不同。'),
('段落 3：人口边界','common500 主图、matched100 的 N2048／敏感性、fixed development 示例与计时分别报告。灰图、回退和失败输出按原协议保留。DISTS、DINO-L、agreement、MS-SSIM 完整结果仍归档；不借删指标制造一致优势。')]),
('III.B. Comparison With Existing Methods',[
('段落 1：主性能','引用 Fig.2，分别描述像素、感知距离与语义表征一致性。保留 NeST 并非所有工作点 PSNR 最高、4 dB LPIPS 不如 Latent JSCC、LPIPS 与 DreamSim 排序存在差异。不能用曲线 CI 重叠判显著性，也不能把 CLIP 差异归因于未经消融的 caption 条件。'),
('段落 2：可视化','引用 Fig.3 的 Pastries95 与 Shopping cart75，描述实际可见的糕点轮廓、表面纹理、购物车网格与把手等；使用统一 ROI 比较局部细节，同时保留完整视野。明确两个案例是既有选定 development 示例，不外推总体优劣。'),
('段落 3：ARPC 解码敏感性（本轮已完成）','matched100、N1024、同一900个接收事件。原采样 τ=.5 复用；显式 argmax 与预定低温 τ=.25 只改变未传尺度生成。19 dB 两种设置四项均改善，贪心 ΔPSNR=+0.150 [0.087,0.207] dB、ΔDreamSim=−0.0143 [−0.0206,−0.0084]；4 dB 两者 LPIPS 和 DreamSim 均变差。10 dB 低温的 PSNR/LPIPS 改善，但贪心四项 CI 跨零。全设置都报告，不用子集替换 common500 主曲线。')]),
('III.C. Effect of Within-Scale Endpoints',[
('段落 1：装载效率','引用 Table I，三个 SNR 的 PARTIAL 利用率均提高；10 dB 从77.70%提高到95.21%。利用率只报告真实均值，不把样本分位数当均值置信区间。'),
('段落 2：同先验机制检验','Δ=PARTIAL−WHOLE。三点 ΔPSNR 约+0.616、+0.701、+0.319 dB，ΔDreamSim 为−0.0207、−0.0240、−0.0030；直接配对区间分别全正／全负。说明合法尺度内端点改善预算匹配，不额外宣称唯一因果机制或新训练收益。')]),
('III.D. Performance With a Larger Symbol Budget',[
('段落 1：有限预算扩展','引用 Table II。matched100、N2048；4/10 dB 为冻结 adaptive，19 dB 为 full representations。每源三噪声，每方法每条件300/300正确送达。表内报告 ΔPSNR 和 ΔDreamSim，均为 NeST−ARPC，不把三行画成同一配置曲线。'),
('段落 2：必须同时保留的边界','19 dB 的 NeST PSNR 更高、DreamSim 更低，但 DISTS 差为+0.0054，95% CI [+0.0024,+0.0086]，ARPC 的 DISTS 更好。保留这一指标取舍，不将 N1024 CLIP 填入 N2048 表。N2048 CLIP 本轮未启动。')]),
('III.E. Computational Cost',[
('段落 1：测量口径','引用 Table III，仅取现有10 dB、16 development、batch1、每源3次预热+3次测量。相同 RTX PRO 6000 Blackwell GPU1 和6个CPU核心；保持原方法环境／精度和共享负载记录，不能声称独占或所有框架完全一致。'),
('段落 2：质量与处理时间','TX/RX/Total 为 software processing latency，单位 ms/image；Total=TX+RX。NeST 均值约220.89/398.82/619.71，连续基线约42–45 ms总计；ARPC约1520.30、BPG约2293.01 ms总计。说明生成式恢复与源码处理的实际成本，不称空口时延或端到端网络时延，不与历史 H800 六方法计时合并。')]),
('IV. Conclusion',[
('段落任务','回扣完整 next-scale token 通信方法：尺度内端点、真实终止码长和前缀约束重建共同适应固定符号预算。总结已观察到的质量与处理成本取舍，并明确低 SNR 感知差异、生成解码敏感性和大预算完整表示边界。不给出全面优于所有方法或所有指标的结论。')])]
write('outline_zh.md','# NeST-Com: Fixed-Budget Next-Scale Token Communication\n\nWCL 修订大纲；以下是段落任务与内容要点，尚非完整论文正文。\n\n'
      '公开基线：`a5f8ceb0233b0e1f599ee8bb902c1a094d47119d`。本轮新增且已完成的科学结果只有 ARPC 解码敏感性；主图均值、CI、示例图和原表数据均复用冻结记录。\n\n'
      +'\n\n'.join('## '+name+'\n\n'+'\n\n'.join('**'+t+'。** '+b for t,b in items) for name,items in sections)
      +'\n\n## 相关工作来源\n\n- Ada-TokenCom: https://arxiv.org/abs/2608.28086\n- TONIC: https://arxiv.org/abs/2605.21553\n\n## 完成状态\n\n三图三表、中文／Word／LaTeX大纲及敏感性结果均属本轮交付；完整论文正文和 N2048 CLIP 不在已执行范围。历史图表和完整八指标结果保留。\n')
(P/'outline_sections.json').write_text(json.dumps(sections,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
write('results_outline.md','# Results paragraph tasks\n\n'+(P/'outline_zh.md').read_text(encoding='utf-8').split('## III.A.')[1].split('## IV.')[0].join(['## III.A.','']))
sr=['# ARPC decoder sensitivity — completed\n',
'The original frozen sampling result is retained. This matched100 analysis changes only generation of untransmitted tokens; it does not replace common500 curves or select a new main checkpoint/policy.\n',
'## Fixed scope and decode semantics\n',
'N=1024; SNR 4, 10 and 19 dB; the original three noise realizations 6201/6202/6203; identical paid caption, MCS, actual received prefixes, weights and source identity. Original sampling uses τ=0.5, CFG=3, top-k=900, top-p=0.97 and the frozen seed 0. The low-temperature value τ=0.25 was fixed before inference. Temperature is passed as a seven-scale vector: 0.5 on received scales, 0.25 only on untransmitted scales. Greedy is an explicit argmax at untransmitted scales; it is not an approximation by an arbitrarily low temperature. Native received-scale sampling/overwrite and state advancement are retained. No TX truth, hidden CDF or unsent token is supplied to RX.\n',
'## Actual work and reuse\n',
'The 900 original events contain 296 distinct received prefixes. The original setting reuses 900 logical results and 296 reconstructions. Two new settings produced 592 unique reconstructions, representing 1800 logical setting-frames with 1208 exact reconstruction reuses. All 592 render calls and all four new score sets completed. There were zero new source encodes, captions, channel events, channel decodes, tokenizer calls, training updates or bootstrap index draws. One preparation syntax error occurred before registration/model calls; it remains recorded. No successful scientific call was replayed.\n',
'RX/CLIP ran on the registered RTX PRO 6000 GPU1 with Python3.12.3/Torch2.10.0+cu130. The native ARPC model/text precision was retained. PSNR, LPIPS-Alex and DreamSim used the frozen H800 metric environment (Python3.10/Torch2.11.0+cu128), original float preprocessing and FP32 metric inference with TF32 disabled. CLIP-I is OpenAI ViT-L/14, frozen 224-pixel float preprocessing and FP32 image–image cosine; 100 exact source embeddings were reused. DreamSim ensemble v0.2.0 LoRA uses the original 224-pixel bicubic/antialiased float preprocessing. Scores come from native reconstruction arrays, not presentation PNGs.\n',
'## Statistics\n',
'Average the three original noises within each source and condition. Each setting/SNR has 100 sources and 300 logical outcomes. Reuse the original 10000×100 bank, seed 2026100701, SHA256 `5e11cd2ee931a2de9f22f8e42034a71ef610eb83a48bac9e53224e666f5a447d`. Form same-source differences first, then resample by the frozen source order. Intervals below are pointwise 95% intervals, with no multiplicity correction. PSNR and CLIP-I favor positive changes; LPIPS and DreamSim favor negative changes. All prespecified settings, including unfavorable and crossing-zero results, are retained.\n',
'## Mean scores\n',
'| Setting | SNR | PSNR | LPIPS-Alex | DreamSim | CLIP-I |\n|---|---:|---:|---:|---:|---:|\n']
rs=rows(D/'summary.csv')
for setting in ['original','greedy','low_tau025']:
 for snr in [4,10,19]:
  vals={r['metric']:float(r['mean']) for r in rs if r['setting']==setting and int(r['snr_db'])==snr}
  sr.append('| '+setting+' | '+str(snr)+' | '+' | '.join(f'{vals[m]:.6f}' for m in ['psnr_db','lpips_alex','dreamsim','clip_image_cosine'])+' |\n')
sr+=['\n## All direct paired differences\n','Δ is the first setting minus the second setting. Full-precision values remain in CSV.\n',
'| Comparison | SNR | Metric | Mean | 95% CI |\n|---|---:|---|---:|---|\n']
for r in rows(D/'paired_differences.csv'):
 sr.append(f"| {r['comparison']} | {r['snr_db']} | {r['metric']} | {float(r['mean']):+.6f} | [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}] |\n")
sr+=['\n## Interpretation for III.B\n',
'At 19 dB, both greedy and τ=0.25 improve all four metrics relative to the original setting, with favorable paired intervals excluding zero. At 4 dB, both alternatives worsen LPIPS and DreamSim; PSNR and CLIP-I intervals cross zero. At 10 dB, low temperature improves PSNR and LPIPS, while its DreamSim and CLIP-I intervals cross zero; all four greedy intervals cross zero. Thus ARPC performance depends on generation decoding and operating point. Do not characterize the original sampler as a universal optimum, and do not silently substitute a subset-selected decoder into the common500 comparison.\n',
'## Evidence and status\n',
'`registration.json` freezes received inputs, settings and RX code; `metric_registration.json` freezes scoring dependencies; `execution_receipts/` preserves actual launches, child waits and completion metadata. The raw closed run is under `liberai-tmp/liulu/liulu-var-comm/arpc_decode_sensitivity_20261021_v1`. RX, three-metric scoring, CLIP scoring and statistics all returned zero; verified science/controller processes are gone and no scientific failure is present. Original arrays, ledgers and complete logs remain on PFS, outside Git. N2048 CLIP was not started.\n']
(S/'REPORT.md').write_text(''.join(sr),encoding='utf-8')
write('README.md',"""# WCL revision package

Title: **NeST-Com: Fixed-Budget Next-Scale Token Communication**

This package revises the outline and actual figures/tables; it is not a full paper draft. The read-only baseline is `a5f8ceb0233b0e1f599ee8bb902c1a094d47119d`. The sole new experiment is the completed, separately registered ARPC decoder sensitivity study.

- `outline_zh.md`: Chinese paragraph tasks; `outline.docx`: the same outline with actual figures/tables.
- `latex/main.tex`: English outline skeleton with actual figures/tables, not full prose.
- `figure1.pptx`: editable modular source reflowed from the user's exact original PPTX; `figure1.pdf/png`: final export.
- `figure2.pdf/svg/png`: 1×4 main comparison, 7.16×2.165 inches.
- `figure3.pdf/svg/png`: selected development examples, complete field of view plus equal per-row ROIs. Native 256×256 pixels are preserved; larger display does not imply additional image detail.
- `table_i–iii.csv/tex`: full-precision data and formatted tables. Table I is now the N1024 mechanism table; Table II is the N2048 paired table; Table III is existing 10 dB software processing latency.
- `captions.md`, `results_outline.md`, `input_manifest.json`, `data_validation.json`, `visual_review.json`: captions, mapping, provenance and checks.
- `sensitivity/REPORT.md` and CSV: all prespecified ARPC settings and direct paired differences. Main common500 curves are unchanged.

The old paired-gain/utilization figure remains archived in v1; its mechanism results now appear in Table I. The previous v2 2×2 performance plot becomes the 1×4 Fig.2. Fig.3 retains the two existing image cases and adds deterministic same-coordinate ROIs. No old figures or user source files were overwritten.

## Reproduction

Run `python draw.py` in this directory using the original NumPy/Matplotlib/Pillow environment. It reads only `inputs/` and checks plotted data and embedded pixels; it performs no models, channel work or bootstrap. `scripts/` contains the deterministic PPTX reflow and size-normalization sources and Word outline builder. Use the bundled artifact-tool runtime for PPTX authoring; final PDF is a native PowerPoint vector export because the bundled direct PDF exporter had an unavailable font alias. No managed package was changed. Font files are not distributed.

Fig.1 and Fig.2 PDFs contain vector artwork/text. Fig.3 necessarily embeds the original raster photographs together with vector annotations. The editable SVG retains the source image pixels and text. See `visual_review.json` for final-size inspections and the deliberate 7 pt minimum for diagram auxiliary text.

N2048 CLIP and complete manuscript prose remain outside this execution. No main quality grid, training, timing, source encoding or channel experiment was rerun.
""")
print('materials written')
