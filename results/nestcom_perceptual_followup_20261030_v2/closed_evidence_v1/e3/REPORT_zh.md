# E3：固定端点上限的粒度控制

100 个固定源、N=1024、10 dB、16-QAM/rate 2/3、共同最大端点 (8,126)、共同逐图噪声及装入规则。保留所有 1,200 个逻辑结果。这里只改变候选粒度；不复用条件不同的旧 Table I 数字。

|候选−基线|指标|均值差|95% 来源配对区间|
|---|---|---:|---|
|quarter−whole|psnr_db|0.65169279|[0.50944486, 0.80673801]|
|quarter−whole|lpips_alex|-0.02749781|[-0.03334443, -0.02194617]|
|quarter−whole|dreamsim|-0.02446709|[-0.03162924, -0.01792041]|
|quarter−whole|clip_image_cosine|0.01677817|[0.00814829, 0.02601540]|
|quarter−whole|dists|-0.01153364|[-0.01410014, -0.00904757]|
|eighth−whole|psnr_db|0.75730266|[0.59558333, 0.93118526]|
|eighth−whole|lpips_alex|-0.03045903|[-0.03671903, -0.02443096]|
|eighth−whole|dreamsim|-0.02505062|[-0.03221046, -0.01845203]|
|eighth−whole|clip_image_cosine|0.01833562|[0.00979059, 0.02735126]|
|eighth−whole|dists|-0.01341913|[-0.01622984, -0.01064633]|
|token−whole|psnr_db|0.82337496|[0.65877119, 1.00100789]|
|token−whole|lpips_alex|-0.03269656|[-0.03923450, -0.02649341]|
|token−whole|dreamsim|-0.02724952|[-0.03418858, -0.02071190]|
|token−whole|clip_image_cosine|0.02339014|[0.01483507, 0.03270810]|
|token−whole|dists|-0.01453607|[-0.01743423, -0.01167179]|
|eighth−quarter|psnr_db|0.10560987|[0.06774477, 0.14656121]|
|eighth−quarter|lpips_alex|-0.00296122|[-0.00407846, -0.00194412]|
|eighth−quarter|dreamsim|-0.00058354|[-0.00200675, 0.00082367]|
|eighth−quarter|clip_image_cosine|0.00155745|[-0.00144987, 0.00455914]|
|eighth−quarter|dists|-0.00188549|[-0.00279185, -0.00106290]|
|token−quarter|psnr_db|0.17168217|[0.12499601, 0.22128561]|
|token−quarter|lpips_alex|-0.00519875|[-0.00653153, -0.00389691]|
|token−quarter|dreamsim|-0.00278243|[-0.00501094, -0.00059336]|
|token−quarter|clip_image_cosine|0.00661197|[0.00173962, 0.01190432]|
|token−quarter|dists|-0.00300242|[-0.00410771, -0.00195442]|
|token−eighth|psnr_db|0.06607230|[0.03356641, 0.10063108]|
|token−eighth|lpips_alex|-0.00223753|[-0.00317602, -0.00131399]|
|token−eighth|dreamsim|-0.00219890|[-0.00422132, -0.00014202]|
|token−eighth|clip_image_cosine|0.00505452|[0.00021935, 0.01050800]|
|token−eighth|dists|-0.00111694|[-0.00203759, -0.00022153]|

PSNR/CLIP-I 正差有利，三个距离负差有利。码长检查次数比较已独立终止的码长，不等于额外先验前向或实测处理时间；计时另见 E5。
