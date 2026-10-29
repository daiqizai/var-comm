# V6 四分贝图稿交付说明

已从实际接收输入补齐固定Pastries95与Dog21的4dB七方法结果。原五方法共十张重建与历史数组哈希逐一相同；新增WHOLE/Ada各两帧及接收重建，未重跑主实验。LPIPS和DreamSim均从原生float32数组计算。

两例NeST相对WHOLE的两项距离均更低，但糕点形状仍被圆化，不能据两张图声称普遍领先。Latent在两图LPIPS及Pastries DreamSim更好；这些结果保留在七方法补充图中。原10dB图也完整保留，正文已有Ada10dB优势未删除。

主稿5页、双倍行距单栏15页、可编辑Word16页、补充材料25页均完成实际渲染检查。Word的长引用网址换行已修复。主定量结果、Fig.1/2、三表与方法方程未修改。新7dB任务单列在results/nestcom_7db_20261030_v1，不将其校准结果作为common500正式性能。

新增计算：WHOLE源码概率遍历2次、prior尺度调用14次；新物理帧4、包译码8；原接收重放10、新WHOLE/Ada接收4；LPIPS14对、DreamSim14对。无新训练、tokenizer/classifier、主研究校准或bootstrap。原失败记录与版本化修复均保留。所有大数组仅留PFS。
