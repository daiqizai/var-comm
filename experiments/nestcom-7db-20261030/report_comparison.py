"""Report displayed, single-realization values; no new scoring/bootstrap."""
from pathlib import Path
import json,csv
W=Path(__file__).resolve().parents[2]
D=W/'results/nestcom_7db_20261030_v1/delivery'
O=W/'results/nestcom_fig3_4db_20261029_v1/delivery'
old=list(csv.DictReader((O/'figure3_metrics.csv').open()))
new=list(csv.DictReader((D/'figure3_metrics.csv').open()))
for r in old:
    if r['method']=='ADA_PREDCLASS_FIXEDFRAME':r['method']='ADA'
rows=[]
for n in new:
    o=next(x for x in old if x['source_id']==n['source_id']and x['method']==n['method'])
    row=dict(source_id=n['source_id'],source_index=n['source_index'],method=n['method'])
    for metric in ['lpips_alex','dreamsim']:
        row[metric+'_4db']=float(o[metric]);row[metric+'_7db']=float(n[metric]);row[metric+'_7_minus_4']=float(n[metric])-float(o[metric])
    rows.append(row)
with(D/'displayed_4_vs_7_metrics.csv').open('x',newline='')as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
account=json.loads((D/'actual_compute_and_reuse.json').read_text())
policy=json.loads((D/'frozen_policies.json').read_text())
text=['# 7 dB 有限校准与固定两图比较','',
'固定 N=1024、Pastries（95）和 Dog（21），比较 WHOLE、NeST、Ada、ARPC、Latent JSCC、SwinJSCC、BPG。每方法每图一个已登记信道实现；本结果为开发集展示，不是 common500 正式新增工作点，也不作总体显著性结论。','',
'## 校准与冻结','',
'NeST、WHOLE、Ada、ARPC 各使用固定100张校准图、五个预定MCS和一次噪声，共2000个实际完整数字帧。按等权来源平均DINOv2余弦选取MCS，失败灰图保留。所有候选结果均在 calibration_summary.csv。BPG精确复用原7 dB、100图×3噪声、PSNR目标的历史冻结策略，因此其校准准则与其他四种数字方法不同。没有根据两张展示图的质量重新选择策略。','',
'| 方法 | 7 dB MCS |','|---|---|']
for method,p in policy['methods'].items():
    q,rate=(p['q'],p['rate'])if'q'in p else p['MCS']
    text.append(f"| {method} | {'QPSK'if q==2 else'16-QAM'}, rate {rate} |")
text+=['','完整帧审计确认所有2000个源码长度均在实际容量内。16-QAM、rate 2/3候选在四种方法中各100/100次拒收；16-QAM、rate 1/2的Ada有1/100次拒收，其余该档方法和三个QPSK候选均成功。失败灰图进入校准分母；此处为100源单次噪声观测，不定位可靠性阈值。','',
'## 同一输出的 LPIPS-Alex / DreamSim（均越低越好）','',
'| 图像 | 方法 | 4 dB | 7 dB |','|---|---|---|---|']
for r in rows:
    text.append(f"| {r['source_index']} | {r['method']} | {r['lpips_alex_4db']:.3f} / {r['dreamsim_4db']:.3f} | {r['lpips_alex_7db']:.3f} / {r['dreamsim_7db']:.3f} |")
text+=['','数值取自实际显示的原生浮点重建，完整精度见 figure3_metrics.csv 与 displayed_4_vs_7_metrics.csv。原方法的噪声种子和采样路径保留，但不宣称跨方法共用噪声向量。7 dB与4 dB采用各自的冻结策略，单图差值也包含策略变化。','',
'## 计算与失败记录','',
f"新校准2000帧，校准独立接收重建{account['new_calibration_received_render']}次，DINO评分{account['calibration_DINO']}对；两图阶段10个数字帧、4个连续JSCC信道试验、14个逻辑重建、14对LPIPS和14对DreamSim。实际GPU总墙时按设备相加为{account['GPU_actual_wall_device_seconds']/3600:.3f}小时，登记上限12设备小时。",
'',
'NeST仅新增2次源码概率遍历；Ada复用精确匹配的类别和独立终止码流；ARPC复用已编码的独立尺度块，并核验原有端点字节一致。新增tokenizer、分类器、训练、bootstrap和common500扩展均为0。初始ARPC环境入口失败发生在子进程启动前，零科学调用；独立环境修订后执行一次并成功，原始失败记录保留。',
'',
'全部阶段的实际退出码、进程身份、调用账本和缓存身份见 actual_compute_and_reuse.json、receipts/ 和 source_receive_reconstruction_map.json。浮点数组保留在PFS，Git仅收轻量记录和展示文件。','']
(D/'REPORT_zh.md').write_text('\n'.join(text),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
