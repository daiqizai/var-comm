# 方法二：模拟残差观测与冻结先验引导

## 四个问题的实际结果

### 1. 较多无噪声残差信息是否有用

g8_c32仍是原16×16 latent的area空间降维观测，并非完整F。以下是其免费正确prefix、无噪声观测的VAR_GUIDED相对各接收控制的真实差值。

| 控制 | 指标 | 差值 [95% CI] | 源数 |
|---|---|---|---:|
| UNGUIDED | psnr_db | 4.5715 [4.2068, 4.9527] | 100 |
| UNGUIDED | lpips_alex | -0.237905 [-0.248638, -0.227465] | 100 |
| UNGUIDED | dino_cosine | 0.366153 [0.33181, 0.401184] | 100 |
| UNGUIDED | dino_specific | 0.366246 [0.328316, 0.404942] | 100 |
| UNGUIDED | F_sq_error_valid | -1350.94 [-1451.29, -1249.55] | 100 |
| LIKELIHOOD | psnr_db | 0.909491 [0.724716, 1.107] | 100 |
| LIKELIHOOD | lpips_alex | -0.0371662 [-0.0419977, -0.0323928] | 100 |
| LIKELIHOOD | dino_cosine | 0.116977 [0.0958728, 0.137128] | 100 |
| LIKELIHOOD | dino_specific | 0.119036 [0.0977073, 0.140417] | 100 |
| LIKELIHOOD | F_sq_error_valid | -167.382 [-196.122, -138.84] | 100 |
| STATIC | psnr_db | 1.28274 [1.06393, 1.52633] | 100 |
| STATIC | lpips_alex | -0.0486836 [-0.0543963, -0.0432786] | 100 |
| STATIC | dino_cosine | 0.131073 [0.10697, 0.156105] | 100 |
| STATIC | dino_specific | 0.13164 [0.105651, 0.158469] | 100 |
| STATIC | F_sq_error_valid | -420.665 [-453.834, -386.464] | 100 |
| DIRECT | psnr_db | 1.20156 [0.994871, 1.4333] | 100 |
| DIRECT | lpips_alex | -0.0502777 [-0.0562952, -0.0445707] | 100 |
| DIRECT | dino_cosine | 0.118474 [0.0953713, 0.142267] | 100 |
| DIRECT | dino_specific | 0.123412 [0.0994347, 0.148426] | 100 |
| DIRECT | F_sq_error_valid | 49.0899 [16.6209, 81.4194] | 100 |

### 2. 降维和加噪后是否仍有来源一致的收益

下面固定7dB，VAR_GUIDED减去同一投影/同一measurement的UNGUIDED。压缩损害另外见`m2_ladder_paired.csv`的无噪声同源比较。

| 投影 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | ΔDINO specificity [95% CI] |
|---|---|---|---|
| g8_c32 | 3.89802 [3.56639, 4.24539] | -0.212685 [-0.222967, -0.202711] | 0.329583 [0.296175, 0.363069] |
| g8_c8 | 2.04386 [1.82612, 2.26913] | -0.112855 [-0.120515, -0.105681] | 0.147075 [0.121638, 0.173164] |
| g6_c8 | 1.41549 [1.22478, 1.61244] | -0.0829698 [-0.0898842, -0.0763058] | 0.124655 [0.101094, 0.148486] |
| g4_c32 | 0.798625 [0.642062, 0.957876] | -0.0763521 [-0.0829845, -0.0700386] | 0.156887 [0.129903, 0.183718] |
| g4_c16 | 0.641487 [0.478349, 0.797556] | -0.05515 [-0.0615057, -0.0489797] | 0.097664 [0.0742532, 0.121102] |
| g4_c8 | 0.496836 [0.367239, 0.622013] | -0.0374699 [-0.0430521, -0.0319261] | 0.0710789 [0.0466708, 0.0952912] |

### 3. 实际链路是否获准并完成

全1000 calibration、7dB、三noise的登记gate状态：**PASSED**。实际链路分支状态：**ACTUAL_LINK_COMPLETE**。

gate在独立analysis中按原数据重算。固定lambda来自前200 calibration，gate包含这200源，因此gate区间是校准内描述性检查，不是独立泛化验证。development未用于调lambda、选投影或改变gate。若gate失败，完整oracle ladder仍发布，实际链路跳过；oracle质量不升级为付费链路结论。

实际投影在前200 calibration按各N/PHY/SNR选择。`selection_fallback=True`表示没有同时满足相对UNGUIDED的PSNR−0.25dB及DINO−0.01约束的候选；报告保留该分支，不把它称为满足质量保护的策略。

### 4. 收益是否特属于VAR先验

下面列出7dB同一观测下VAR_GUIDED对LIKELIHOOD、STATIC、DIRECT的LPIPS差区间。VAR仅胜UNGUIDED不足以证明先验机制；需同时检查这些控制以及PSNR、DINO specificity、F误差。

| 投影 | 对照 | ΔLPIPS [95% CI] | 区间支持LPIPS下降 |
|---|---|---|---|
| g8_c32 | LIKELIHOOD | -0.0414149 [-0.0457249, -0.0373529] | 是 |
| g8_c32 | STATIC | -0.0485818 [-0.0535171, -0.0437679] | 是 |
| g8_c32 | DIRECT | -0.0352101 [-0.0400637, -0.0304622] | 是 |
| g4_c16 | LIKELIHOOD | -0.123101 [-0.135266, -0.111535] | 是 |
| g4_c16 | STATIC | -0.101909 [-0.116582, -0.0879769] | 是 |
| g4_c16 | DIRECT | -0.0263003 [-0.0320138, -0.0208265] | 是 |
| g4_c32 | LIKELIHOOD | -0.107064 [-0.118486, -0.0962025] | 是 |
| g4_c32 | STATIC | -0.0985964 [-0.112331, -0.0857588] | 是 |
| g4_c32 | DIRECT | -0.0369172 [-0.0428736, -0.0312131] | 是 |
| g4_c8 | LIKELIHOOD | -0.118095 [-0.132492, -0.104433] | 是 |
| g4_c8 | STATIC | -0.113719 [-0.128286, -0.0998583] | 是 |
| g4_c8 | DIRECT | -0.018275 [-0.0230958, -0.0133935] | 是 |
| g6_c8 | LIKELIHOOD | -0.0954708 [-0.10685, -0.0847237] | 是 |
| g6_c8 | STATIC | -0.0946442 [-0.10697, -0.0829002] | 是 |
| g6_c8 | DIRECT | -0.0418254 [-0.0476857, -0.0362021] | 是 |
| g8_c8 | LIKELIHOOD | -0.0906075 [-0.100184, -0.0812798] | 是 |
| g8_c8 | STATIC | -0.0821858 [-0.0920881, -0.0729638] | 是 |
| g8_c8 | DIRECT | -0.0555989 [-0.0617679, -0.0497158] | 是 |

## 实际付费链路的配对结果

各接收控制共享同一实际波形、观测和事件；历史P只共享源与名义seed。以下每个区间均来自100源内平均三noise后的配对差。

| N | PHY | SNR | 投影 | 对照 | 指标 | 差值 [95% CI] |
|---:|---|---:|---|---|---|---|
| 1024 | 16QAM | 1 | g4_c16 | UNGUIDED | psnr_db | 0.00823792 [-0.0185255, 0.0365923] |
| 1024 | 16QAM | 1 | g4_c16 | UNGUIDED | lpips_alex | -0.00166459 [-0.00288736, -0.000697076] |
| 1024 | 16QAM | 1 | g4_c16 | UNGUIDED | dino_specific | 0.00251626 [-0.000748282, 0.00639895] |
| 1024 | 16QAM | 1 | g4_c16 | LIKELIHOOD | psnr_db | 0.0220147 [-0.0181225, 0.0807707] |
| 1024 | 16QAM | 1 | g4_c16 | LIKELIHOOD | lpips_alex | -0.00503228 [-0.00911859, -0.00188221] |
| 1024 | 16QAM | 1 | g4_c16 | LIKELIHOOD | dino_specific | 0.017158 [0.007065, 0.0302683] |
| 1024 | 16QAM | 1 | g4_c16 | STATIC | psnr_db | 0.0441766 [0.00172989, 0.103606] |
| 1024 | 16QAM | 1 | g4_c16 | STATIC | lpips_alex | -0.00421954 [-0.00828141, -0.00121813] |
| 1024 | 16QAM | 1 | g4_c16 | STATIC | dino_specific | 0.0146594 [0.00565321, 0.0263769] |
| 1024 | 16QAM | 1 | g4_c16 | DIRECT | psnr_db | -0.0079056 [-0.0360217, 0.0160597] |
| 1024 | 16QAM | 1 | g4_c16 | DIRECT | lpips_alex | -0.000350523 [-0.00111371, 0.000284288] |
| 1024 | 16QAM | 1 | g4_c16 | DIRECT | dino_specific | 0.00158359 [-0.00192609, 0.00543429] |
| 512 | 16QAM | 1 | g4_c16 | UNGUIDED | psnr_db | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | UNGUIDED | lpips_alex | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | UNGUIDED | dino_specific | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | LIKELIHOOD | psnr_db | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | LIKELIHOOD | lpips_alex | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | LIKELIHOOD | dino_specific | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | STATIC | psnr_db | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | STATIC | lpips_alex | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | STATIC | dino_specific | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | DIRECT | psnr_db | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | DIRECT | lpips_alex | 0 [0, 0] |
| 512 | 16QAM | 1 | g4_c16 | DIRECT | dino_specific | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | UNGUIDED | psnr_db | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | UNGUIDED | lpips_alex | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | UNGUIDED | dino_specific | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | LIKELIHOOD | psnr_db | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | LIKELIHOOD | lpips_alex | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | LIKELIHOOD | dino_specific | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | STATIC | psnr_db | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | STATIC | lpips_alex | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | STATIC | dino_specific | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | DIRECT | psnr_db | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | DIRECT | lpips_alex | 0 [0, 0] |
| 512 | QPSK | 1 | g4_c16 | DIRECT | dino_specific | 0 [0, 0] |
| 512 | 16QAM | 4 | g4_c16 | UNGUIDED | psnr_db | 0.126999 [0.0782077, 0.176481] |
| 512 | 16QAM | 4 | g4_c16 | UNGUIDED | lpips_alex | -0.0117685 [-0.0150621, -0.00868235] |
| 512 | 16QAM | 4 | g4_c16 | UNGUIDED | dino_specific | 0.0206568 [0.0111726, 0.0311165] |
| 512 | 16QAM | 4 | g4_c16 | LIKELIHOOD | psnr_db | 0.0561298 [-0.0210795, 0.149054] |
| 512 | 16QAM | 4 | g4_c16 | LIKELIHOOD | lpips_alex | -0.0242476 [-0.0310867, -0.018074] |
| 512 | 16QAM | 4 | g4_c16 | LIKELIHOOD | dino_specific | 0.0730736 [0.053254, 0.0943137] |
| 512 | 16QAM | 4 | g4_c16 | STATIC | psnr_db | 0.140851 [0.0638979, 0.231382] |
| 512 | 16QAM | 4 | g4_c16 | STATIC | lpips_alex | -0.0196138 [-0.0262168, -0.0138708] |
| 512 | 16QAM | 4 | g4_c16 | STATIC | dino_specific | 0.0655626 [0.0469113, 0.0862075] |
| 512 | 16QAM | 4 | g4_c16 | DIRECT | psnr_db | 0.0441301 [-0.0028755, 0.0904519] |
| 512 | 16QAM | 4 | g4_c16 | DIRECT | lpips_alex | -0.00549441 [-0.00748032, -0.00362021] |
| 512 | 16QAM | 4 | g4_c16 | DIRECT | dino_specific | 0.0149877 [0.00710392, 0.023265] |
| 512 | QPSK | 4 | g4_c16 | UNGUIDED | psnr_db | 0.371061 [0.266372, 0.479429] |
| 512 | QPSK | 4 | g4_c16 | UNGUIDED | lpips_alex | -0.0337727 [-0.0392078, -0.0285731] |
| 512 | QPSK | 4 | g4_c16 | UNGUIDED | dino_specific | 0.0562081 [0.040005, 0.0736414] |
| 512 | QPSK | 4 | g4_c16 | LIKELIHOOD | psnr_db | 0.26067 [0.109889, 0.433611] |
| 512 | QPSK | 4 | g4_c16 | LIKELIHOOD | lpips_alex | -0.0749569 [-0.0847432, -0.06568] |
| 512 | QPSK | 4 | g4_c16 | LIKELIHOOD | dino_specific | 0.234112 [0.201795, 0.268189] |
| 512 | QPSK | 4 | g4_c16 | STATIC | psnr_db | 0.517488 [0.35625, 0.701544] |
| 512 | QPSK | 4 | g4_c16 | STATIC | lpips_alex | -0.0633338 [-0.0737784, -0.0532332] |
| 512 | QPSK | 4 | g4_c16 | STATIC | dino_specific | 0.213706 [0.18253, 0.246741] |
| 512 | QPSK | 4 | g4_c16 | DIRECT | psnr_db | 0.132628 [0.0330908, 0.231707] |
| 512 | QPSK | 4 | g4_c16 | DIRECT | lpips_alex | -0.0158601 [-0.0202663, -0.0117664] |
| 512 | QPSK | 4 | g4_c16 | DIRECT | dino_specific | 0.0380591 [0.0229082, 0.054433] |
| 1024 | QPSK | 1 | g6_c8 | UNGUIDED | psnr_db | 0.0478374 [0.00871079, 0.0911707] |
| 1024 | QPSK | 1 | g6_c8 | UNGUIDED | lpips_alex | -0.00180087 [-0.00358678, -0.00010701] |
| 1024 | QPSK | 1 | g6_c8 | UNGUIDED | dino_specific | 0.00207812 [-0.00183405, 0.00607093] |
| 1024 | QPSK | 1 | g6_c8 | LIKELIHOOD | psnr_db | 0.0389331 [0.0123603, 0.0726381] |
| 1024 | QPSK | 1 | g6_c8 | LIKELIHOOD | lpips_alex | -0.00663948 [-0.0114667, -0.00286857] |
| 1024 | QPSK | 1 | g6_c8 | LIKELIHOOD | dino_specific | 0.0200536 [0.00988613, 0.0316832] |
| 1024 | QPSK | 1 | g6_c8 | STATIC | psnr_db | 0.0388261 [0.014211, 0.0696362] |
| 1024 | QPSK | 1 | g6_c8 | STATIC | lpips_alex | -0.00708779 [-0.0122227, -0.00303261] |
| 1024 | QPSK | 1 | g6_c8 | STATIC | dino_specific | 0.0167136 [0.00794567, 0.026967] |
| 1024 | QPSK | 1 | g6_c8 | DIRECT | psnr_db | -0.00190159 [-0.0408593, 0.031469] |
| 1024 | QPSK | 1 | g6_c8 | DIRECT | lpips_alex | -0.000132838 [-0.00155639, 0.00140692] |
| 1024 | QPSK | 1 | g6_c8 | DIRECT | dino_specific | 0.000475047 [-0.00326408, 0.003991] |
| 512 | QPSK | 13 | g6_c8 | UNGUIDED | psnr_db | 1.59695 [1.3991, 1.79699] |
| 512 | QPSK | 13 | g6_c8 | UNGUIDED | lpips_alex | -0.0925437 [-0.0999182, -0.0853918] |
| 512 | QPSK | 13 | g6_c8 | UNGUIDED | dino_specific | 0.135923 [0.113031, 0.159478] |
| 512 | QPSK | 13 | g6_c8 | LIKELIHOOD | psnr_db | 0.551883 [0.380516, 0.738024] |
| 512 | QPSK | 13 | g6_c8 | LIKELIHOOD | lpips_alex | -0.0958742 [-0.106852, -0.0857277] |
| 512 | QPSK | 13 | g6_c8 | LIKELIHOOD | dino_specific | 0.370597 [0.340132, 0.401507] |
| 512 | QPSK | 13 | g6_c8 | STATIC | psnr_db | 0.714299 [0.53563, 0.912414] |
| 512 | QPSK | 13 | g6_c8 | STATIC | lpips_alex | -0.0922491 [-0.104073, -0.0811052] |
| 512 | QPSK | 13 | g6_c8 | STATIC | dino_specific | 0.349469 [0.317417, 0.382407] |
| 512 | QPSK | 13 | g6_c8 | DIRECT | psnr_db | 0.544649 [0.38939, 0.707894] |
| 512 | QPSK | 13 | g6_c8 | DIRECT | lpips_alex | -0.050679 [-0.0572178, -0.0443408] |
| 512 | QPSK | 13 | g6_c8 | DIRECT | dino_specific | 0.103647 [0.0808033, 0.127261] |
| 512 | QPSK | 19 | g6_c8 | UNGUIDED | psnr_db | 1.67583 [1.47336, 1.88226] |
| 512 | QPSK | 19 | g6_c8 | UNGUIDED | lpips_alex | -0.0957007 [-0.10249, -0.0891112] |
| 512 | QPSK | 19 | g6_c8 | UNGUIDED | dino_specific | 0.144655 [0.11875, 0.170311] |
| 512 | QPSK | 19 | g6_c8 | LIKELIHOOD | psnr_db | 0.594248 [0.425163, 0.781162] |
| 512 | QPSK | 19 | g6_c8 | LIKELIHOOD | lpips_alex | -0.0968212 [-0.107976, -0.0865715] |
| 512 | QPSK | 19 | g6_c8 | LIKELIHOOD | dino_specific | 0.366817 [0.335138, 0.397707] |
| 512 | QPSK | 19 | g6_c8 | STATIC | psnr_db | 0.698821 [0.518276, 0.900789] |
| 512 | QPSK | 19 | g6_c8 | STATIC | lpips_alex | -0.0905971 [-0.102685, -0.0791709] |
| 512 | QPSK | 19 | g6_c8 | STATIC | dino_specific | 0.356322 [0.323447, 0.389189] |
| 512 | QPSK | 19 | g6_c8 | DIRECT | psnr_db | 0.616734 [0.459289, 0.778875] |
| 512 | QPSK | 19 | g6_c8 | DIRECT | lpips_alex | -0.0537082 [-0.0596231, -0.0478985] |
| 512 | QPSK | 19 | g6_c8 | DIRECT | dino_specific | 0.111485 [0.0864195, 0.137203] |
| 512 | 16QAM | 7 | g6_c8 | UNGUIDED | psnr_db | 1.40505 [1.21079, 1.60182] |
| 512 | 16QAM | 7 | g6_c8 | UNGUIDED | lpips_alex | -0.0843312 [-0.0909145, -0.0780984] |
| 512 | 16QAM | 7 | g6_c8 | UNGUIDED | dino_specific | 0.130262 [0.106081, 0.154592] |
| 512 | 16QAM | 7 | g6_c8 | LIKELIHOOD | psnr_db | 0.55756 [0.395698, 0.735963] |
| 512 | 16QAM | 7 | g6_c8 | LIKELIHOOD | lpips_alex | -0.0988991 [-0.109684, -0.0888859] |
| 512 | 16QAM | 7 | g6_c8 | LIKELIHOOD | dino_specific | 0.376054 [0.343759, 0.408444] |
| 512 | 16QAM | 7 | g6_c8 | STATIC | psnr_db | 0.66735 [0.500024, 0.852875] |
| 512 | 16QAM | 7 | g6_c8 | STATIC | lpips_alex | -0.0949629 [-0.107309, -0.0833393] |
| 512 | 16QAM | 7 | g6_c8 | STATIC | dino_specific | 0.347015 [0.316069, 0.378359] |
| 512 | 16QAM | 7 | g6_c8 | DIRECT | psnr_db | 0.389069 [0.247271, 0.534514] |
| 512 | 16QAM | 7 | g6_c8 | DIRECT | lpips_alex | -0.0434677 [-0.0490649, -0.0379619] |
| 512 | 16QAM | 7 | g6_c8 | DIRECT | dino_specific | 0.0999028 [0.0761755, 0.123442] |
| 512 | QPSK | 7 | g6_c8 | UNGUIDED | psnr_db | 1.40689 [1.21409, 1.60414] |
| 512 | QPSK | 7 | g6_c8 | UNGUIDED | lpips_alex | -0.0844616 [-0.0910246, -0.0783357] |
| 512 | QPSK | 7 | g6_c8 | UNGUIDED | dino_specific | 0.130445 [0.106074, 0.154771] |
| 512 | QPSK | 7 | g6_c8 | LIKELIHOOD | psnr_db | 0.556849 [0.393591, 0.735613] |
| 512 | QPSK | 7 | g6_c8 | LIKELIHOOD | lpips_alex | -0.0991447 [-0.110056, -0.0891047] |
| 512 | QPSK | 7 | g6_c8 | LIKELIHOOD | dino_specific | 0.378537 [0.345561, 0.411904] |
| 512 | QPSK | 7 | g6_c8 | STATIC | psnr_db | 0.669725 [0.501849, 0.855468] |
| 512 | QPSK | 7 | g6_c8 | STATIC | lpips_alex | -0.0951079 [-0.107407, -0.083383] |
| 512 | QPSK | 7 | g6_c8 | STATIC | dino_specific | 0.349051 [0.316968, 0.38142] |
| 512 | QPSK | 7 | g6_c8 | DIRECT | psnr_db | 0.386769 [0.244169, 0.532574] |
| 512 | QPSK | 7 | g6_c8 | DIRECT | lpips_alex | -0.0434705 [-0.0490788, -0.0379358] |
| 512 | QPSK | 7 | g6_c8 | DIRECT | dino_specific | 0.100016 [0.0760576, 0.123693] |
| 1024 | 16QAM | 13 | g8_c8 | UNGUIDED | psnr_db | 2.2653 [2.04063, 2.50402] |
| 1024 | 16QAM | 13 | g8_c8 | UNGUIDED | lpips_alex | -0.121947 [-0.129721, -0.114482] |
| 1024 | 16QAM | 13 | g8_c8 | UNGUIDED | dino_specific | 0.179621 [0.153506, 0.205873] |
| 1024 | 16QAM | 13 | g8_c8 | LIKELIHOOD | psnr_db | 0.590128 [0.428606, 0.769501] |
| 1024 | 16QAM | 13 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0878033 [-0.0971693, -0.0789136] |
| 1024 | 16QAM | 13 | g8_c8 | LIKELIHOOD | dino_specific | 0.357195 [0.327567, 0.387778] |
| 1024 | 16QAM | 13 | g8_c8 | STATIC | psnr_db | 0.533412 [0.37715, 0.711982] |
| 1024 | 16QAM | 13 | g8_c8 | STATIC | lpips_alex | -0.0813301 [-0.0912067, -0.0719238] |
| 1024 | 16QAM | 13 | g8_c8 | STATIC | dino_specific | 0.33519 [0.304903, 0.366072] |
| 1024 | 16QAM | 13 | g8_c8 | DIRECT | psnr_db | 0.726965 [0.585058, 0.874353] |
| 1024 | 16QAM | 13 | g8_c8 | DIRECT | lpips_alex | -0.0639181 [-0.0699259, -0.0581902] |
| 1024 | 16QAM | 13 | g8_c8 | DIRECT | dino_specific | 0.131219 [0.107516, 0.154998] |
| 1024 | QPSK | 13 | g8_c8 | UNGUIDED | psnr_db | 2.2653 [2.04063, 2.50402] |
| 1024 | QPSK | 13 | g8_c8 | UNGUIDED | lpips_alex | -0.121947 [-0.129721, -0.114482] |
| 1024 | QPSK | 13 | g8_c8 | UNGUIDED | dino_specific | 0.179621 [0.153506, 0.205873] |
| 1024 | QPSK | 13 | g8_c8 | LIKELIHOOD | psnr_db | 0.590128 [0.428606, 0.769501] |
| 1024 | QPSK | 13 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0878033 [-0.0971693, -0.0789136] |
| 1024 | QPSK | 13 | g8_c8 | LIKELIHOOD | dino_specific | 0.357195 [0.327567, 0.387778] |
| 1024 | QPSK | 13 | g8_c8 | STATIC | psnr_db | 0.533412 [0.37715, 0.711982] |
| 1024 | QPSK | 13 | g8_c8 | STATIC | lpips_alex | -0.0813301 [-0.0912067, -0.0719238] |
| 1024 | QPSK | 13 | g8_c8 | STATIC | dino_specific | 0.33519 [0.304903, 0.366072] |
| 1024 | QPSK | 13 | g8_c8 | DIRECT | psnr_db | 0.726965 [0.585058, 0.874353] |
| 1024 | QPSK | 13 | g8_c8 | DIRECT | lpips_alex | -0.0639181 [-0.0699259, -0.0581902] |
| 1024 | QPSK | 13 | g8_c8 | DIRECT | dino_specific | 0.131219 [0.107516, 0.154998] |
| 512 | 16QAM | 13 | g8_c8 | UNGUIDED | psnr_db | 2.20809 [1.98757, 2.43392] |
| 512 | 16QAM | 13 | g8_c8 | UNGUIDED | lpips_alex | -0.121937 [-0.12942, -0.11483] |
| 512 | 16QAM | 13 | g8_c8 | UNGUIDED | dino_specific | 0.163559 [0.138253, 0.188864] |
| 512 | 16QAM | 13 | g8_c8 | LIKELIHOOD | psnr_db | 0.52871 [0.392548, 0.675444] |
| 512 | 16QAM | 13 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0879286 [-0.0972827, -0.0792505] |
| 512 | 16QAM | 13 | g8_c8 | LIKELIHOOD | dino_specific | 0.351561 [0.317928, 0.384981] |
| 512 | 16QAM | 13 | g8_c8 | STATIC | psnr_db | 0.486771 [0.334505, 0.654549] |
| 512 | 16QAM | 13 | g8_c8 | STATIC | lpips_alex | -0.0804092 [-0.0909871, -0.0708554] |
| 512 | 16QAM | 13 | g8_c8 | STATIC | dino_specific | 0.314972 [0.283156, 0.346586] |
| 512 | 16QAM | 13 | g8_c8 | DIRECT | psnr_db | 0.67881 [0.534922, 0.822295] |
| 512 | 16QAM | 13 | g8_c8 | DIRECT | lpips_alex | -0.0641668 [-0.0700508, -0.0585275] |
| 512 | 16QAM | 13 | g8_c8 | DIRECT | dino_specific | 0.115123 [0.0918239, 0.138935] |
| 1024 | 16QAM | 19 | g8_c8 | UNGUIDED | psnr_db | 2.31422 [2.07102, 2.57462] |
| 1024 | 16QAM | 19 | g8_c8 | UNGUIDED | lpips_alex | -0.124946 [-0.133247, -0.117108] |
| 1024 | 16QAM | 19 | g8_c8 | UNGUIDED | dino_specific | 0.170045 [0.142475, 0.197366] |
| 1024 | 16QAM | 19 | g8_c8 | LIKELIHOOD | psnr_db | 0.593575 [0.437687, 0.767548] |
| 1024 | 16QAM | 19 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0885257 [-0.0984518, -0.0792696] |
| 1024 | 16QAM | 19 | g8_c8 | LIKELIHOOD | dino_specific | 0.346447 [0.313739, 0.378707] |
| 1024 | 16QAM | 19 | g8_c8 | STATIC | psnr_db | 0.522727 [0.367851, 0.699651] |
| 1024 | 16QAM | 19 | g8_c8 | STATIC | lpips_alex | -0.07914 [-0.0879688, -0.0707869] |
| 1024 | 16QAM | 19 | g8_c8 | STATIC | dino_specific | 0.314296 [0.283561, 0.344728] |
| 1024 | 16QAM | 19 | g8_c8 | DIRECT | psnr_db | 0.764941 [0.605308, 0.928437] |
| 1024 | 16QAM | 19 | g8_c8 | DIRECT | lpips_alex | -0.0667511 [-0.0732177, -0.0605207] |
| 1024 | 16QAM | 19 | g8_c8 | DIRECT | dino_specific | 0.121716 [0.0969155, 0.145882] |
| 1024 | QPSK | 19 | g8_c8 | UNGUIDED | psnr_db | 2.31422 [2.07102, 2.57462] |
| 1024 | QPSK | 19 | g8_c8 | UNGUIDED | lpips_alex | -0.124946 [-0.133247, -0.117108] |
| 1024 | QPSK | 19 | g8_c8 | UNGUIDED | dino_specific | 0.170045 [0.142475, 0.197366] |
| 1024 | QPSK | 19 | g8_c8 | LIKELIHOOD | psnr_db | 0.593575 [0.437687, 0.767548] |
| 1024 | QPSK | 19 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0885257 [-0.0984518, -0.0792696] |
| 1024 | QPSK | 19 | g8_c8 | LIKELIHOOD | dino_specific | 0.346447 [0.313739, 0.378707] |
| 1024 | QPSK | 19 | g8_c8 | STATIC | psnr_db | 0.522727 [0.367851, 0.699651] |
| 1024 | QPSK | 19 | g8_c8 | STATIC | lpips_alex | -0.07914 [-0.0879688, -0.0707869] |
| 1024 | QPSK | 19 | g8_c8 | STATIC | dino_specific | 0.314296 [0.283561, 0.344728] |
| 1024 | QPSK | 19 | g8_c8 | DIRECT | psnr_db | 0.764941 [0.605308, 0.928437] |
| 1024 | QPSK | 19 | g8_c8 | DIRECT | lpips_alex | -0.0667511 [-0.0732177, -0.0605207] |
| 1024 | QPSK | 19 | g8_c8 | DIRECT | dino_specific | 0.121716 [0.0969155, 0.145882] |
| 512 | 16QAM | 19 | g8_c8 | UNGUIDED | psnr_db | 2.27636 [2.05349, 2.50479] |
| 512 | 16QAM | 19 | g8_c8 | UNGUIDED | lpips_alex | -0.125101 [-0.132697, -0.117709] |
| 512 | 16QAM | 19 | g8_c8 | UNGUIDED | dino_specific | 0.170001 [0.144858, 0.194832] |
| 512 | 16QAM | 19 | g8_c8 | LIKELIHOOD | psnr_db | 0.537858 [0.391988, 0.695081] |
| 512 | 16QAM | 19 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0886792 [-0.0980887, -0.0797886] |
| 512 | 16QAM | 19 | g8_c8 | LIKELIHOOD | dino_specific | 0.346895 [0.314386, 0.379166] |
| 512 | 16QAM | 19 | g8_c8 | STATIC | psnr_db | 0.505589 [0.367398, 0.656202] |
| 512 | 16QAM | 19 | g8_c8 | STATIC | lpips_alex | -0.0818527 [-0.0920014, -0.0724715] |
| 512 | 16QAM | 19 | g8_c8 | STATIC | dino_specific | 0.318076 [0.286431, 0.349604] |
| 512 | 16QAM | 19 | g8_c8 | DIRECT | psnr_db | 0.732064 [0.594751, 0.871602] |
| 512 | 16QAM | 19 | g8_c8 | DIRECT | lpips_alex | -0.0670531 [-0.0730856, -0.061189] |
| 512 | 16QAM | 19 | g8_c8 | DIRECT | dino_specific | 0.121264 [0.0982032, 0.143786] |
| 1024 | 16QAM | 4 | g8_c8 | UNGUIDED | psnr_db | 1.29185 [1.11443, 1.47625] |
| 1024 | 16QAM | 4 | g8_c8 | UNGUIDED | lpips_alex | -0.0686169 [-0.0752557, -0.0620768] |
| 1024 | 16QAM | 4 | g8_c8 | UNGUIDED | dino_specific | 0.0933902 [0.0738062, 0.112842] |
| 1024 | 16QAM | 4 | g8_c8 | LIKELIHOOD | psnr_db | 0.449025 [0.344562, 0.567107] |
| 1024 | 16QAM | 4 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0651356 [-0.0736458, -0.0567996] |
| 1024 | 16QAM | 4 | g8_c8 | LIKELIHOOD | dino_specific | 0.259776 [0.230287, 0.290262] |
| 1024 | 16QAM | 4 | g8_c8 | STATIC | psnr_db | 0.393211 [0.284616, 0.516227] |
| 1024 | 16QAM | 4 | g8_c8 | STATIC | lpips_alex | -0.0577975 [-0.0661241, -0.0498361] |
| 1024 | 16QAM | 4 | g8_c8 | STATIC | dino_specific | 0.239233 [0.208963, 0.270468] |
| 1024 | 16QAM | 4 | g8_c8 | DIRECT | psnr_db | 0.284284 [0.189529, 0.380609] |
| 1024 | 16QAM | 4 | g8_c8 | DIRECT | lpips_alex | -0.0290736 [-0.0336684, -0.0246276] |
| 1024 | 16QAM | 4 | g8_c8 | DIRECT | dino_specific | 0.0608171 [0.0446622, 0.0767553] |
| 1024 | QPSK | 4 | g8_c8 | UNGUIDED | psnr_db | 1.2964 [1.11914, 1.48214] |
| 1024 | QPSK | 4 | g8_c8 | UNGUIDED | lpips_alex | -0.0688406 [-0.0754827, -0.0622808] |
| 1024 | QPSK | 4 | g8_c8 | UNGUIDED | dino_specific | 0.0942136 [0.0743988, 0.113757] |
| 1024 | QPSK | 4 | g8_c8 | LIKELIHOOD | psnr_db | 0.452079 [0.34669, 0.570646] |
| 1024 | QPSK | 4 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0655067 [-0.0741278, -0.0571391] |
| 1024 | QPSK | 4 | g8_c8 | LIKELIHOOD | dino_specific | 0.260922 [0.231435, 0.291359] |
| 1024 | QPSK | 4 | g8_c8 | STATIC | psnr_db | 0.396221 [0.287625, 0.518489] |
| 1024 | QPSK | 4 | g8_c8 | STATIC | lpips_alex | -0.0580433 [-0.0663669, -0.050026] |
| 1024 | QPSK | 4 | g8_c8 | STATIC | dino_specific | 0.239904 [0.209791, 0.2712] |
| 1024 | QPSK | 4 | g8_c8 | DIRECT | psnr_db | 0.282953 [0.18771, 0.378802] |
| 1024 | QPSK | 4 | g8_c8 | DIRECT | lpips_alex | -0.0291366 [-0.0337284, -0.0247213] |
| 1024 | QPSK | 4 | g8_c8 | DIRECT | dino_specific | 0.0615562 [0.0451999, 0.0775495] |
| 1024 | 16QAM | 7 | g8_c8 | UNGUIDED | psnr_db | 2.08778 [1.86244, 2.31777] |
| 1024 | 16QAM | 7 | g8_c8 | UNGUIDED | lpips_alex | -0.111281 [-0.119131, -0.103883] |
| 1024 | 16QAM | 7 | g8_c8 | UNGUIDED | dino_specific | 0.154205 [0.128535, 0.180308] |
| 1024 | 16QAM | 7 | g8_c8 | LIKELIHOOD | psnr_db | 0.631339 [0.478368, 0.80727] |
| 1024 | 16QAM | 7 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0885743 [-0.0988256, -0.0791649] |
| 1024 | 16QAM | 7 | g8_c8 | LIKELIHOOD | dino_specific | 0.354049 [0.322776, 0.385535] |
| 1024 | 16QAM | 7 | g8_c8 | STATIC | psnr_db | 0.592451 [0.436649, 0.768853] |
| 1024 | 16QAM | 7 | g8_c8 | STATIC | lpips_alex | -0.0827191 [-0.0934295, -0.072885] |
| 1024 | 16QAM | 7 | g8_c8 | STATIC | dino_specific | 0.33445 [0.304057, 0.365203] |
| 1024 | 16QAM | 7 | g8_c8 | DIRECT | psnr_db | 0.593366 [0.450561, 0.738084] |
| 1024 | 16QAM | 7 | g8_c8 | DIRECT | lpips_alex | -0.0539602 [-0.0601529, -0.0480511] |
| 1024 | 16QAM | 7 | g8_c8 | DIRECT | dino_specific | 0.105748 [0.0830537, 0.128359] |
| 1024 | QPSK | 7 | g8_c8 | UNGUIDED | psnr_db | 2.08778 [1.86244, 2.31777] |
| 1024 | QPSK | 7 | g8_c8 | UNGUIDED | lpips_alex | -0.111281 [-0.119131, -0.103883] |
| 1024 | QPSK | 7 | g8_c8 | UNGUIDED | dino_specific | 0.154205 [0.128535, 0.180308] |
| 1024 | QPSK | 7 | g8_c8 | LIKELIHOOD | psnr_db | 0.631339 [0.478368, 0.80727] |
| 1024 | QPSK | 7 | g8_c8 | LIKELIHOOD | lpips_alex | -0.0885743 [-0.0988256, -0.0791649] |
| 1024 | QPSK | 7 | g8_c8 | LIKELIHOOD | dino_specific | 0.354049 [0.322776, 0.385535] |
| 1024 | QPSK | 7 | g8_c8 | STATIC | psnr_db | 0.592451 [0.436649, 0.768853] |
| 1024 | QPSK | 7 | g8_c8 | STATIC | lpips_alex | -0.0827191 [-0.0934295, -0.072885] |
| 1024 | QPSK | 7 | g8_c8 | STATIC | dino_specific | 0.33445 [0.304057, 0.365203] |
| 1024 | QPSK | 7 | g8_c8 | DIRECT | psnr_db | 0.593366 [0.450561, 0.738084] |
| 1024 | QPSK | 7 | g8_c8 | DIRECT | lpips_alex | -0.0539602 [-0.0601529, -0.0480511] |
| 1024 | QPSK | 7 | g8_c8 | DIRECT | dino_specific | 0.105748 [0.0830537, 0.128359] |
| 1024 | 16QAM | 1 | g4_c16 | P1024 | psnr_db | -4.01861 [-4.27614, -3.76314] |
| 1024 | 16QAM | 1 | g4_c16 | P1024 | lpips_alex | 0.145743 [0.135356, 0.15655] |
| 1024 | 16QAM | 1 | g4_c16 | P1024 | dino_specific | 0.0340566 [-0.00302401, 0.0712511] |
| 512 | 16QAM | 1 | g4_c16 | P512 | psnr_db | -7.92381 [-8.33015, -7.52221] |
| 512 | 16QAM | 1 | g4_c16 | P512 | lpips_alex | 0.36386 [0.347678, 0.38038] |
| 512 | 16QAM | 1 | g4_c16 | P512 | dino_specific | -0.155789 [-0.179354, -0.132362] |
| 512 | QPSK | 1 | g4_c16 | P512 | psnr_db | -7.50057 [-7.8639, -7.14923] |
| 512 | QPSK | 1 | g4_c16 | P512 | lpips_alex | 0.334422 [0.318159, 0.351148] |
| 512 | QPSK | 1 | g4_c16 | P512 | dino_specific | -0.12619 [-0.152687, -0.0992825] |
| 512 | 16QAM | 4 | g4_c16 | P512 | psnr_db | -4.92851 [-5.23058, -4.63274] |
| 512 | 16QAM | 4 | g4_c16 | P512 | lpips_alex | 0.17279 [0.160216, 0.185815] |
| 512 | 16QAM | 4 | g4_c16 | P512 | dino_specific | 0.0767146 [0.0418839, 0.112656] |
| 512 | QPSK | 4 | g4_c16 | P512 | psnr_db | -3.7239 [-3.9943, -3.45115] |
| 512 | QPSK | 4 | g4_c16 | P512 | lpips_alex | 0.0987671 [0.0894703, 0.108072] |
| 512 | QPSK | 4 | g4_c16 | P512 | dino_specific | 0.194463 [0.157719, 0.230921] |
| 1024 | QPSK | 1 | g6_c8 | P1024 | psnr_db | -3.94284 [-4.20922, -3.68017] |
| 1024 | QPSK | 1 | g6_c8 | P1024 | lpips_alex | 0.142771 [0.133008, 0.153009] |
| 1024 | QPSK | 1 | g6_c8 | P1024 | dino_specific | 0.0392665 [0.00326036, 0.0755556] |
| 512 | QPSK | 13 | g6_c8 | P512 | psnr_db | -3.4799 [-3.71618, -3.24316] |
| 512 | QPSK | 13 | g6_c8 | P512 | lpips_alex | 0.0916363 [0.0840383, 0.0991896] |
| 512 | QPSK | 13 | g6_c8 | P512 | dino_specific | 0.169176 [0.132445, 0.206538] |
| 512 | QPSK | 19 | g6_c8 | P512 | psnr_db | -3.54462 [-3.77756, -3.31156] |
| 512 | QPSK | 19 | g6_c8 | P512 | lpips_alex | 0.0950301 [0.087381, 0.102779] |
| 512 | QPSK | 19 | g6_c8 | P512 | dino_specific | 0.161458 [0.124788, 0.197977] |
| 512 | 16QAM | 7 | g6_c8 | P512 | psnr_db | -3.15753 [-3.37727, -2.93942] |
| 512 | 16QAM | 7 | g6_c8 | P512 | lpips_alex | 0.0737481 [0.0662195, 0.0813602] |
| 512 | 16QAM | 7 | g6_c8 | P512 | dino_specific | 0.217789 [0.18204, 0.254195] |
| 512 | QPSK | 7 | g6_c8 | P512 | psnr_db | -3.15304 [-3.37371, -2.93422] |
| 512 | QPSK | 7 | g6_c8 | P512 | lpips_alex | 0.0736875 [0.0661808, 0.0813272] |
| 512 | QPSK | 7 | g6_c8 | P512 | dino_specific | 0.218325 [0.182481, 0.254884] |
| 1024 | 16QAM | 13 | g8_c8 | P1024 | psnr_db | -3.86657 [-4.09562, -3.63599] |
| 1024 | 16QAM | 13 | g8_c8 | P1024 | lpips_alex | 0.125652 [0.11735, 0.133914] |
| 1024 | 16QAM | 13 | g8_c8 | P1024 | dino_specific | -0.062102 [-0.0917705, -0.0336455] |
| 1024 | QPSK | 13 | g8_c8 | P1024 | psnr_db | -3.86657 [-4.09562, -3.63599] |
| 1024 | QPSK | 13 | g8_c8 | P1024 | lpips_alex | 0.125652 [0.11735, 0.133914] |
| 1024 | QPSK | 13 | g8_c8 | P1024 | dino_specific | -0.062102 [-0.0917705, -0.0336455] |
| 512 | 16QAM | 13 | g8_c8 | P512 | psnr_db | -2.86876 [-3.07523, -2.66093] |
| 512 | 16QAM | 13 | g8_c8 | P512 | lpips_alex | 0.0622428 [0.0558382, 0.0685762] |
| 512 | 16QAM | 13 | g8_c8 | P512 | dino_specific | 0.196811 [0.162432, 0.231389] |
| 1024 | 16QAM | 19 | g8_c8 | P1024 | psnr_db | -3.99547 [-4.23456, -3.75657] |
| 1024 | 16QAM | 19 | g8_c8 | P1024 | lpips_alex | 0.128898 [0.120846, 0.136889] |
| 1024 | 16QAM | 19 | g8_c8 | P1024 | dino_specific | -0.0890546 [-0.120554, -0.0588807] |
| 1024 | QPSK | 19 | g8_c8 | P1024 | psnr_db | -3.99547 [-4.23456, -3.75657] |
| 1024 | QPSK | 19 | g8_c8 | P1024 | lpips_alex | 0.128898 [0.120846, 0.136889] |
| 1024 | QPSK | 19 | g8_c8 | P1024 | dino_specific | -0.0890546 [-0.120554, -0.0588807] |
| 512 | 16QAM | 19 | g8_c8 | P512 | psnr_db | -2.94409 [-3.15577, -2.73455] |
| 512 | 16QAM | 19 | g8_c8 | P512 | lpips_alex | 0.0656301 [0.0590614, 0.0721735] |
| 512 | 16QAM | 19 | g8_c8 | P512 | dino_specific | 0.186804 [0.151574, 0.221987] |
| 1024 | 16QAM | 4 | g8_c8 | P1024 | psnr_db | -3.62946 [-3.86612, -3.39865] |
| 1024 | 16QAM | 4 | g8_c8 | P1024 | lpips_alex | 0.124933 [0.114769, 0.135365] |
| 1024 | 16QAM | 4 | g8_c8 | P1024 | dino_specific | -0.00158593 [-0.0348705, 0.0308478] |
| 1024 | QPSK | 4 | g8_c8 | P1024 | psnr_db | -3.62296 [-3.85962, -3.39185] |
| 1024 | QPSK | 4 | g8_c8 | P1024 | lpips_alex | 0.124687 [0.114555, 0.135143] |
| 1024 | QPSK | 4 | g8_c8 | P1024 | dino_specific | -0.000689251 [-0.0340619, 0.0317964] |
| 1024 | 16QAM | 7 | g8_c8 | P1024 | psnr_db | -3.45691 [-3.66798, -3.24668] |
| 1024 | 16QAM | 7 | g8_c8 | P1024 | lpips_alex | 0.111783 [0.104443, 0.119341] |
| 1024 | 16QAM | 7 | g8_c8 | P1024 | dino_specific | -0.0215384 [-0.0493206, 0.00523372] |
| 1024 | QPSK | 7 | g8_c8 | P1024 | psnr_db | -3.45691 [-3.66798, -3.24668] |
| 1024 | QPSK | 7 | g8_c8 | P1024 | lpips_alex | 0.111783 [0.104443, 0.119341] |
| 1024 | QPSK | 7 | g8_c8 | P1024 | dino_specific | -0.0215384 [-0.0493206, 0.00523372] |

[实际链路质量曲线](figures/m2_actual_quality_vs_SNR.png)。16QAM质量比较按相同N和调制，实际E在汇总表单独列出，不能称严格同E。

## 独立计时

每context/control先warmup，再对10个固定源各计时2次；先平均同源两次再平均10源。oracle endpoint假设正确prefix/gain，其速度不能当部署链路速度。实际endpoint包含付费帧收发和真实失败路径；固定算子缓存允许，图像输出缓存禁用。

[分范围TX/RX/总时间](m2_timing_summary.csv)。

## 预算和oracle边界

六投影的noiseless/noisy oracle均假设正确数字m4前缀与正确gain；noisy观测在相应归一化波形上真实加AWGN。g8_c32的完整观测无法放入N512/N1024付费链路。其他投影的可行尺寸由`m2_resource_ledger.csv`列出，尺寸可行仍不等于真实header/CRC/gain可靠。

与冻结P512/P1024的源配对差值见`m2_P_paired.csv`。oracle行明确`fair_paid_link_comparison=False`，仅作观测诊断；实际链路才按相应N对比。P的历史namespace不同，名义seed相同不代表同一物理noise。共同错图DINO reference采用历史derangement。

似然使用登记的Gaussian working/composite approximation、未观测残差统计及固定前向算子。没有声明精确独立后验、后验采样、全球最优或无推理成本。模型参数训练更新为0；calibration统计/PCA和输入变量迭代属于额外数据/算力使用。

## 统计、分母与证据

清洁观测每源1次seed0；noisy每源三noise平均，再对100源做10,000次共享bootstrap，seed20261002。所有失败图像留在质量分母；实际有效latent的F误差和双方共同有效F配对另注明分母。

- [汇总](m2_summary.csv)、[逐源均值](m2_source_means.csv)、[控制配对](m2_paired_intervals.csv)、[压缩损害](m2_ladder_paired.csv)、[P比较](m2_P_paired.csv)。
- [全量校准gate独立复算](m2_gate_independent_audit.json)、[gate区间](m2_gate_independent_paired.csv)。
- [维度与质量](figures/m2_oracle_damage_ladder.png)、[有噪观测曲线](figures/m2_oracle_quality_vs_SNR.png)。
- 固定样例source_index=0/25/50/75、clean/4/13dB、seed0/2001，目录`m2_examples/`。

本报告只确认方法二分析阶段。整个研究和推送状态由supervisor/publication证据确认。
