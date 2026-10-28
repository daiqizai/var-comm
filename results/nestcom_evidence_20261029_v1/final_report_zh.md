# 本轮交付报告：NeST-Com evidence 2026-10-29
## 新实验
E1/E2已真实闭合。E1在固定100源、3工作点×3原重复下比较原Ada-sampling、新Ada-argmax和既有NeST，共2700逻辑记录。E2在同100源、N5124/10dB×3原重复比较R0/R1，共1200记录。全部四指标、灰图和分母保留。

- E1：N5124/10dB，argmax−sampling PSNR为−0.475[−0.801,−0.152]／−0.651[−0.984,−0.338]dB；NeST−argmax为+4.632[4.183,5.093]／+4.600[4.188,5.018]dB。CLIP对照区间跨零，不能称全指标领先。N102410dB，NeST−argmax PSNR+0.568[0.246,0.902]dB，但DreamSim+0.0178[0.0129,0.0229]及CLIP仍有利Ada。
- E2：补全−粗尺度直接解码PSNR在4dB为+0.026[−0.204,0.271]dB、10dB为+0.771[0.563,0.986]dB；两个点LPIPS/DreamSim/CLIP均改善。正文保留4dB的PSNR不确定性。

实际新增：E1唯一反事实285、资格回放4；E2唯一反事实200、资格回放4；新render/decoder共493。E1priorforward73984，E2仅资格回放40（R0自身0）。新四指标各485对，复用原sourcefeature；GPU占用累计0.798732设备小时，含已保留的CLIP模型前失败时间。新PHY、训练、源码编码、tokenizer、bootstrap抽样和figure_restore均为0。E1复用1845逻辑输出，E2复用603。账目不把逻辑结果冒称新模型调用。

## 论文与图表
新稿沿用指定标题、匿名作者，保留N1024/N512/粒度/A0B0C0主干。关键E1/E2数值及CI直接进入III.B/C/D；六系统取舍、不利结果、DINO校准关联、独立预算校准与SNR8dB全拒收均明确。无全指标SOTA、Quarter最优、逐图单调、SNR鲁棒或比Ada更快主张。

Fig1原PPTX三面板与鸟图/网格保留，百分比删除为示意，双重作用明确；PPTX可编辑，SVG文字转路径。Fig2保留36个原均值和36CI。Fig3为N51210dB2×4、宽6inch高3.30inch，原糕点/狗身份不在正式common500，按已登记规则取manifest前两源；仅第一重复6201，逐图指标，不以质量择图。旧N1024图完整保留为补充。

双栏PDF5页10pt，摘要按实际规则计数见data_checks；双倍行距单栏PDF15页，Word实际15页。三表为原生Word表，123个可编辑OMML对象，正文未栅格化。全部35页实际查看或经逐像素一致确认承接已查看版面；新布局不沿用旧PASS。编译返回码0，内置LaTeX编译器也成功。Word渲染工具因无LibreOffice失败后使用安装的Word16只读导出，记录原因。

## 交付入口与限制
`results/nestcom_evidence_20261029_v1/e1/`、`e2/`含完整frame/source/summary/pairedCSV及报告；`actual_compute_and_reuse.json`、`statistics_identity.json`、各注册和闭合记录提供证据。
`paper/manuscripts/nestcom_wcl_evidence_20261029_v1/`含TeX、两PDF、Word、三图各格式、三表CSV/TeX、来源映射、主张矩阵和验收。
科学输入大数组/权重/完整日志仍在PFS，未迁移、未上传Git。既有停止分支未恢复。E1/E2是固定既有子集诊断，不是独立盲测；完整系统比较不能分离所有模型因素；E2不能排除既有Dc适配。

Git仅远端原仓库main，基准`8bab318fbc8f8f115e96c9aeffeef10d3c2bb4b6`；精确发布结果以`validation/publication.json`为准。当前报告不冒称未执行的仓库检查/推送已通过。
