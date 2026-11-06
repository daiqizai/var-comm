"""Fixed 500-source paired bootstrap; no checkpoint or hyperparameter selection."""
import csv,json,time
from pathlib import Path
import numpy as np
from runtime import *
from formal_codec_v1 import METRICS,CAPACITIES,digest

def csvfile(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    F=L/'formal500_v1';r=json.loads((F/'protocol.json').read_text());out=F/'delivery';out.mkdir(exist_ok=False)
    order=[v['source_id'] for v in json.loads(Path(r['manifest']).read_text())['sources']]
    ident=json.loads(Path(r['bank_identity']).read_text());assert ident['sources']==order and len(order)==500
    assert sha(r['bank'])==r['bank_sha256'];bank=np.load(r['bank'],allow_pickle=False)
    assert bank.shape==(10000,500) and digest(bank)==ident['array_sha256']
    rows=[];counts={};ledger=[]
    for folder in [F/'qualification']+[F/f'shard{i}' for i in range(8)]:
        assert json.loads((folder/'complete.json').read_text())['passed']
        for stage in ['tx','rx','score_v2']:
            a=json.loads((folder/'roles'/stage/'actual_wait.json').read_text());assert a['returncode']==0 and a['child_gone']
            ledger.append(dict(role=folder.name+'/'+stage,**a))
            c=json.loads((folder/stage/'complete.json').read_text())['counts']
            for k,v in c.items():counts[('qualification_' if folder.name=='qualification' else 'formal_')+k]=counts.get(('qualification_' if folder.name=='qualification' else 'formal_')+k,0)+v
        if folder.name!='qualification':rows.extend(csv.DictReader((folder/'score_v2/frame_metrics.csv').open()))
    assert len(rows)==5000
    key={(v['source'],v['method'],v['condition']):v for v in rows};assert len(key)==5000
    methods=['original_nest','trained5000'];conditions=[str(c) for c in CAPACITIES]+['full']
    rows=[key[s,m,c] for s in order for m in methods for c in conditions]
    summary=[];paired=[];capacity=[]
    def ci(a):
        means=np.concatenate([a[bank[i:i+500]].mean(1) for i in range(0,10000,500)])
        lo,hi=np.quantile(means,[.025,.975]);return float(lo),float(hi)
    for c in conditions:
        values={}
        for m in methods:
            items=[key[s,m,c] for s in order]
            for metric in METRICS:
                v=np.array([float(x[metric]) for x in items]);assert np.isfinite(v).all();values[m,metric]=v
                lo,hi=ci(v);summary.append(dict(method=m,condition=c,metric=metric,n_sources=500,mean=float(v.mean()),ci_low=lo,ci_high=hi))
            bits=np.array([int(x['source_bits']) for x in items]);tokens=np.array([int(x['tokens']) for x in items])
            if c!='full':
                assert all(int(x['source_bits'])<=int(c) and int(x['packet_bits'])==int(x['source_bits'])+41<=int(c)+41 for x in items)
            else:assert np.all(tokens==680)
            capacity.append(dict(method=m,condition=c,n_sources=500,mean_source_bits=float(bits.mean()),min_source_bits=int(bits.min()),max_source_bits=int(bits.max()),
                mean_tokens=float(tokens.mean()),null_count=sum(x['null']=='True' for x in items),
                mean_source_utilization=float(bits.mean()/int(c)) if c!='full' else '',full_streams_exceeding_13bit_length=int((bits>=8192).sum()) if c=='full' else ''))
        for metric in METRICS:
            d=values['trained5000',metric]-values['original_nest',metric];lo,hi=ci(d)
            paired.append(dict(comparison='trained5000_minus_original_nest',condition=c,metric=metric,n_sources=500,mean_delta=float(d.mean()),ci_low=lo,ci_high=hi,
                favorable_direction='positive' if metric in ['psnr_db','clip_i'] else 'negative'))
    csvfile(out/'frame_metrics.csv',rows);csvfile(out/'source_means.csv',rows);csvfile(out/'summary.csv',summary);csvfile(out/'paired_deltas.csv',paired);csvfile(out/'capacity.csv',capacity)
    failed=json.loads((F/'qualification/roles/score/actual_wait.json').read_text());assert failed['returncode']==1 and failed['child_gone']
    ledger.append(dict(role='qualification/score_import_failure',**failed))
    gpu=sum(v['charged_gpu_device_hours'] for v in ledger);cpu=sum(v['cpu_process_seconds'] for v in ledger)/3600
    assert gpu<=r['gpu_device_hour_limit'] and cpu<=r['cpu_process_hour_limit']
    write(out/'actual_compute_and_reuse.json',dict(counts=counts,logical_results=5000,new_wireless_frames=0,
        source_population='fixed500_COCO_holdout_not_claimed_WCL_common500',native_float_scoring=True,old_jobs_paused=True))
    write(out/'resource_ledger.json',dict(gpu_device_hours=gpu,cpu_process_hours=cpu,stages=ledger))
    write(out/'audit.json',dict(passed=True,checkpoint_step=5000,sources=500,logical_results=5000,mean_CI=50,paired_CI=25,
        bank_sha256=r['bank_sha256'],all_received_prefixes_exact=True,all_finite=True,full_tokens=680,clean_link_only=True,
        selection='fixed final step5000; no test-driven selection',time=time.time()))
    lines=['# 原始 NeST 率失真迁移：固定 5000 步正式测试','',
        '固定 500 张 COCO 留出图；与训练、开发和资格图的原生像素身份无交集。原始 NeST 权重与最终 5000 步权重使用相同的真实算术编码器、651 个合法端点、逐源最长可装入前缀及独立接收流程。',
        '四档实际源码容量为 591 / 1273 / 1911 / 2548 bit，长度与 CRC 共 29 bit，另付 12 bit profile；总信息包上限为 632 / 1314 / 1952 / 2589 bit。本轮为无噪声源码包比较，没有新增 LDPC/QAM/无线帧，也不复用 RDVQ 的基线质量。',
        '完整表示单独独立解码全部 680 tokens，不施加上述预算，报告实际终止码长；它不是等比特率工作点。模型、种子和测试样本均未按测试结果重选。',
        '五指标为 PSNR、LPIPS-Alex、DISTS、DreamSim、CLIP-I；训练开发使用的 LPIPS-VGG 不与本次 LPIPS-Alex 数值混用。来源 bootstrap 复用原 10000×500 固定索引及源顺序，区间由直接逐源差值计算。','',
        '| 条件 | 指标 | 新−原均值 | 95% 来源配对区间 |','|---|---|---:|---|']
    for v in paired:lines.append(f"| {v['condition']} | {v['metric']} | {v['mean_delta']:+.6f} | [{v['ci_low']:+.6f}, {v['ci_high']:+.6f}] |")
    lines+=['',f'实际本次 GPU 设备小时 {gpu:.6f}；CPU 进程小时 {cpu:.6f}。详细调用与复用次数见 actual_compute_and_reuse.json，原四臂和旧大模型继续暂停。',
        '这些结果衡量整套联合适配后的变化，不能单独归因于可微率项；没有额外的同训练量去率项对照。所有混合或不利结果均保留。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
