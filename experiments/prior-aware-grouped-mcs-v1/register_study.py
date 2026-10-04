"""Freeze the reviewed queue and report only already verified Stage A facts."""
from pathlib import Path
import argparse
import gzip
import shutil
from uep_common import read,write,seal,sha,require
from release_tables import copy_light

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();out=a.out.resolve();code=Path(__file__).resolve().parent
    qualification=read(out/'ldpc_qualification.json');enumeration=read(out/'stage_a/resource_enumeration.json')
    require(qualification['status']=='PASS' and enumeration['encoder_qualified'],'Actual backend engineering qualification required')
    native=read(root/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/delivery/config.json')
    share=['--policies','{POLICIES}','--quality-bundle','{QUALITY_BUNDLE}','--quality-completion','{QUALITY_COMPLETION}',
           '--bler','{BLER}','--codebook','{CODEBOOK}']
    commands=[
      dict(name='P1024',gpu=True,completion_relative='p1024_scores/completion.json',args=['{CODE}/p1024_driver.py','--root','{ROOT}','--output','{OUT}/p1024_scores',
            '--convnext-weights','{OUT}/weights/convnext_tiny-983f1562.pth',*share]),
      dict(name='receiver_cost',gpu=True,completion_relative='receiver_cost/timing.json',args=['{CODE}/receiver_cost.py','--root','{ROOT}','--policies','{POLICIES}',
            '--events','{OUT}/actual_events','--output','{OUT}/receiver_cost']),
      dict(name='model_validation',completion_relative='model_validation.json',args=['{CODE}/model_validation.py','--policies','{POLICIES}','--bler','{BLER}',
            '--events','{OUT}/actual_events','--scores','{OUT}/actual_scores','--preregistration','{OUT}/owner_config_r2.json','--output','{OUT}/model_validation.json']),
      dict(name='extension_gate',completion_relative='extension_gate.json',args=['{CODE}/decide_extension.py','--policies','{POLICIES}',
            '--scores','{OUT}/actual_scores','--validation','{OUT}/model_validation.json','--output','{OUT}/extension_gate.json']),
      dict(name='report',completion_relative='../../results/prior_aware_uep_20261004/publication_manifest.json',args=['{CODE}/report_builder.py','--root','{ROOT}',
            '--scored','{OUT}/actual_scores','--p-scored','{OUT}/p1024_scores','--policies','{POLICIES}',
            '--model-validation','{OUT}/model_validation.json','--gate','{OUT}/extension_gate.json','--timing','{OUT}/receiver_cost/timing.json',
            '--fixed-examples','{ROOT}/outputs/EXTERNAL-COMPARISON-20261004/fixed_examples.json']),
      dict(name='prepare_delivery',completion_relative='publish_manifest.json',args=['{CODE}/prepare_delivery.py','--root','{ROOT}','--out','{OUT}']),
      dict(name='publish',completion_relative='publication_completion.json',args=['{CODE}/publish.py','--root','{ROOT}','--out','{OUT}',
            '--manifest','{OUT}/publish_manifest.json','--message','Report prior-aware UEP N1024 actual-link evaluation and independent validation'])]
    for command in commands:require((code/Path(command['args'][0]).name).is_file(),'Missing registered delivery implementation')
    upstream=root/'outputs/M1-N2048-FULL-GRID-20261004-FAST-R1'
    cfg=dict(study_id='PRIOR-AWARE-UEP-20261004-V1',execution_revision='r2_plain_publication_tables',root=str(root),out=str(out),native_python=native['native_python'],
        native_environment=native['native_environment'],source_bindings={str(f):sha(f) for f in sorted(code.iterdir()) if f.is_file()},
        upstream_completion=str(upstream/'publication_completion.json'),upstream_required_status='PUSHED_AND_STOPPED',
        upstream_failure_files=[str(upstream/n) for n in ('owner_failure.json','delivery_failure.json','failure.json')],
        coarse_initial_upper_hours=4.,refinement_allowance_hours=4.,actual_allowance_hours=6.,
        allowances_are_planning_not_measurements=True,quality_cost_pilot_pending=True,delivery_commands=commands,
        training_updates=0,calibration_selection_metric='dinov2_vitl14_cosine',development_read=False,
        independent_validation='convnext_source_prediction_agreement',N2048_UEP_extension='conditional registered gate; no automatic scientific success claim')
    seal(out/'owner_config_r2.json',cfg)
    result=root/'results/prior_aware_uep_20261004_stage_a';result.mkdir(parents=True,exist_ok=True)
    for name in ('ldpc_qualification.json','owner_config_r2.json','environment_completion.json','ldpc_requirements_lock.txt','coarse_concurrency_revision.json','CPU_thread_priority_revision.json'):
        if (out/name).exists():shutil.copyfile(out/name,result/name)
    for name in ('resource_enumeration.json','K_generation.json','candidate_profiles_N1024.json','candidate_profiles_N2048.json','resource_ledger.csv'):
        copy_light(out/'stage_a'/name,result/name)
    tests=(out/'cpu_tests_final.log').read_text();require('failed' not in tests.lower() and 'passed' in tests,'Reviewed CPU tests have not passed')
    (result/'cpu_tests.txt').write_text(tests,encoding='utf-8')
    text='''# 先验感知 UEP：登记与阶段 A

本轮已启动 CPU 实链路测表；还没有新的图像质量结论。现有 M1 N2048 全动作空间评测继续运行。

## 已核验

- Sionna 2.2.0 的实际 5G LDPC 编解码可用。6 种代表配置完成无噪声往返，CRC、68 次付费头与 QPSK/16QAM 映射通过核验。
- N1024 完整枚举得到 4,041 个合法候选、3,761 个不同波形配置、147 个恢复状态。m9 等不可行状态按实际预算排除，没有降低 B0 的合法搜索范围。
- 去重后的正文测表含 2,816 个物理配置；六档 SNR、每点256码块，加头部共4,326,912码块。先粗测，入围点再按100错块或20,000块精化。
- CPU 实测选用32帧批量。八分片保持原随机计数器，当前四路并行、每路两线程、降低CPU调度优先级，避免拖慢已有GPU推断。先前八路的测量记录和安全检查点保留。
- 新 LDPC 环境独立安装，没有修改现有模型环境。CPU 测表禁用 CUDA；没有新训练或开发集选策。

## 七项补充已登记

1. 原 VAR 是无类别 class1000 的确定性逐尺度 argmax。本次固定同一规则，Q 表与真实接收不使用随机 token 采样。
2. DINOv2 ViT-L/14 用于校准选策；独立 ConvNeXt-Tiny V1 原图预测一致率只在策略冻结后评测。
3. B3−B0 须在至少两个 SNR 上同时通过两个指标的正向配对区间，且模型验证通过、B3未退化为单组，才具备 N2048 UEP 复核资格。区间跨零不追加样本。
4. 登记 P1024 固定40k的10dB推断，以及同源4/7/13dB的严格缓存复用或原 latent 重解码。
5. 单独测不使用最终图像缓存的熵序发送端和接收端耗时；历史60–75ms只作历史参照，不代替新测量。
6. GPU 先等待现有 M1 N2048 交付，再做两张校准图的全状态测速。默认1000张完整搜索；按事先固定的48小时规则，超时则300张筛选、1000张复核。后者不声称全1000全网格最优。
7. 使用实际 Sionna LDPC；没有把卷积码正文冒充LDPC。共享头部沿用原卷积保护并完整计费。

## 排程与时间边界

质量表与真实链路还未执行，完整完工时间尚不能精确给出。CPU粗表的首轮八路观测折算约1.2小时；现改四路低优先级，暂按2–4小时安排，按实际进度更新。GPU成本需要两源试跑后才能确认，不能把CPU速度套到生成和全指标评分上。

旧校准结果没有保存浮点重建图，且缺DINO-L等指标，因此不直接用于新协议选策。新Q按147个状态复用，不按4,041个候选重复生成；300→1000复核复用自身已封印的图像和指标。

连续可信前缀接收丢弃CRC失败组；旧失败硬候选不是新B0。16QAM仅称同平均功率，逐帧实际能量单列。所有开发比较固定原100图×3噪声，不打开holdout。

代码与协议：[执行入口](../experiments/prior-aware-grouped-mcs-v1/README.md)。资源、资格与队列登记：[阶段A结果](../results/prior_aware_uep_20261004_stage_a/)。
'''
    report=root/'reports/prior_aware_uep_stage_a_20261004.md';report.write_text(text,encoding='utf-8')
    note='\n\n2026-10-04: Registered prior-aware UEP N1024 with real Sionna5G LDPC, continuous accepted-prefix RX, independent ConvNeXt validation and conditional N2048 replication. CPU BLER shards running; GPU source-Q waits for existing M1 N2048 delivery. No new quality result or training. See [Stage A](reports/prior_aware_uep_stage_a_20261004.md).\n'
    for name in ('PROGRESS.md','RESEARCH_STATUS.md','EXPERIMENTS.md'):
        path=root/name;old=path.read_text()
        if note.strip() not in old:path.write_text(old+note,encoding='utf-8')
    files=[f for f in code.iterdir() if f.is_file()]+[f for f in result.rglob('*') if f.is_file()]+[report,*[root/n for n in ('PROGRESS.md','RESEARCH_STATUS.md','EXPERIMENTS.md')]]
    require(all(f.stat().st_size<10_000_000 for f in files),'Initial publication file too large')
    init=out/'initial_release_r2';init.mkdir(exist_ok=True)
    value=dict(publish_paths=[str(f.relative_to(root)) for f in files],file_sha256={str(f.relative_to(root)):sha(f) for f in files},
        stage='ENGINEERING_AND_EXECUTION_REGISTRATION_ONLY',scientific_quality_results=False)
    seal(init/'publish_manifest.json',value);print('Frozen queue and',len(files),'initial publication files')

if __name__=='__main__':main()
