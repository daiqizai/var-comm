# 新诊断的解释与限制
E1：N512的argmax并没有改善原Ada采样结果。4/10dB的PSNR下降0.475/0.651dB，三个感知/特征指标也变差。NeST−argmax仍为4.632/4.600dB，LPIPS/DreamSim有利NeST，CLIP区间跨零。在N102410dB，argmax−sampling的PSNR区间跨零；NeST虽有约0.568dB优势，但DreamSim和CLIP仍有利Ada。原sampling主曲线保留；不能把这些差异全部归因于next-scale端点。

E2：在同一实收WHOLE前缀上，补全在N5124dB的PSNR增益0.026dB，区间[-0.204,0.271]跨零；10dB为0.771[0.563,0.986]dB。LPIPS/DreamSim/CLIP在两个点均改善。结论是适配固定接收机中的补全作用具有指标和工作点差异，不能声称强先验普遍提高源图PSNR，也不能排除既有Dc适配因素。

全部四指标和三个E1/两个E2工作点均报告。固定100源是既有子集，不是新独立盲测；先平均3原重复，再做等权来源配对。原10000×100索引仅复用重采样身份，不继承旧ARPC比较描述。没有相减边际CI、500源均值或切片500源bank。

E1保留原NeST灰图2个逻辑事件，Ada0；E2每臂4dB灰图3个、10dB0。所有分母不变。每个工作点/臂仍100源×3重复。新反事实不是新增无线帧。原生数组路径/hash在frame_metrics.csv中，图像PNG只展示。
