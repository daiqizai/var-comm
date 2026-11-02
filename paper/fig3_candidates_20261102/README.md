# Fig.3 全部领先候选：2026-11-02

基准 main `f03a295ed956b2a6dd376a032f088a5eaa7936f9`。当前稿 Fig.3 为 N1024、7 dB、Source / WHOLE / Ada / NeST。当前电视源图 84 满足五项均领先。

本轮读取既有完整候选池（7/10 dB 各 500 源），保留原第一信道重复。筛选条件为 NeST 的 DISTS、DreamSim 均严格小于 WHOLE 与 Ada；另标记五项指标全部领先。

7 dB：143 个双指标候选，60 个五指标候选，58 个五指标且无灰图候选。3 个双指标候选含对照灰图，完整保留并显式标注。
10 dB：23 个双指标候选，11 个五指标候选，均无灰图。

默认 HTML 展示 7 dB 五项均领先且无灰图；切换筛选即可看完整 166 个候选。PDF 包含全部双指标候选，7 dB 36 页、10 dB 6 页；五项领先、无灰图优先，随后保留原优势排序。编号关联完整 source ID；可据编号选图。

所有图像保持完整 256×256 视野，精确提取自既有哈希固定的 contact sheets；22 页原接触图复用、664 张显示图片、零新模型/PHY/指标/统计调用。已通过源图像素身份、完整源标识、参考 hash、方法、重复与实际输出身份连接；原电视/皮革两例的提取 RGB 与既有展示图片逐字节相同。显示 PNG 不替代原生数组指标输入。

这里是按优势筛选的插图候选；正式 common500 均值、CI、失败分母及当前稿件未改动。五项领先是当前显示实现逐图数值比较，不是逐图统计显著性判定。

文件：`index.html` 为可筛选图库；两个 PDF 为全部候选；`all_166_leading_candidates.csv` 为候选数值；`all_1000_candidates_audit.csv` 同时保留未入选工作点；`candidate_metadata.json` / `extraction_identity.json` 保存来源身份。

## 存放与重现

当前活动稿 `paper/manuscripts/nestcom_wcl_completion_20261101_v9/` 未修改；本目录仅为选图目录。完整图库、两份 PDF、664 张显示 PNG 与 CSV 保存在 PFS `.../liulu-var-comm/fig3_candidates_20261102/review_delivery.zip` 和用户 Downloads 的同名候选目录。Git 只保存脚本、轻量 CSV/来源记录及第一页预览，原生数组不入库。

重现：下载并解压完整交付，使用 `scripts/build_catalog.py --input-dir <交付目录>/inputs --tiles-zip <PFS>/fig3_candidates_20261102/display_tiles.zip --output-dir <全新输出目录>`。依赖 Python、Pillow、PyMuPDF、ReportLab；当前字体为 Windows Arial，可按平台替换等宽度可读字体后检查版面。

展示采用 PDF 技能的实际渲染验收：42 页均渲染，7 张全页布局总览覆盖全部页面并实际打开，另逐页打开 8 张首/中/末页检查字号与图像对应。HTML 文件的筛选成员计数已静态核对；内置浏览器不允许 file URL，未宣称已完成浏览器端视觉测试。可直接使用 PDF 审阅。
