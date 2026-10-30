"""No retuning: apply the registered development screening and tie rules."""
from opt_common import *
import csv

def main(role,route):
 folder=D/(role+'_delivery');g=read(folder/'gate.json');gates=g['gates'];candidates=[]
 for name,gate in gates.items():
  if route=='A'and name!='A1':continue
  ok=gate['fit_pass']
  if route=='A':ok=ok and gate['verification_seed_direction_replicated']
  if route=='E':ok=ok and gate['core_normalized_means']['lpips_alex']<=.99 and gate['core_normalized_means']['dreamsim']<=1
  if ok:candidates.append(name)
 selected=None
 if candidates:
  best=min(gates[n]['score']for n in candidates);tied=[n for n in candidates if gates[n]['score']-best<.001]
  if route=='B':tied.sort(key=lambda n:float(n.split('T')[1]))
  elif route=='D':tied.sort(key=lambda n:0 if n=='D_eighth'else 1)
  else:tied.sort()
  selected=tied[0]
 guards_done=selected is not None and gates[selected].get('guard_cells_complete',role=='fit_A')
 status='NO_GO'if selected is None else'PREPARED'
 record=dict(route=route,stage='FIT100',status=status,first_round_science_complete=True,selected=selected,guards_complete=guards_done,next_required='None; do not promote this route'if selected is None else'Frozen CONFIRM100 evaluation'if guards_done else'Same frozen candidate on five guard cells before CONFIRM',gate=pin(folder/'gate.json'),automatic_hyperparameter_search=False,primary_A_seed=2026102201 if route=='A'else None,verification_A_seed_not_selected=True if route=='A'else None)
 save(folder/'decision.json',record)
 lines=[f'路线 {route}：首轮 FIT100 已完成，筛选状态 {status}。','此结果来自冻结的开发子集，不是独立最终验证。','每个来源先平均生成重复、再平均信道重复；区间直接按来源配对重采样。']
 for name,v in gates.items():lines.append(f"{name}: J={v['score']:.9f}; 核心归一化 LPIPS={v['core_normalized_means']['lpips_alex']:.9f}, DreamSim={v['core_normalized_means']['dreamsim']:.9f}; 护栏违反={json.dumps(v['guard_violations'],ensure_ascii=False)}")
 if selected is None:lines.append('没有通过预定条件；保留全部正负指标，不进入该路线确认，不把验证种子改作优胜配置。')
 else:lines.append('冻结候选：'+selected+'；仍需完成上述护栏/独立来源确认，不能据此宣称最终改善。')
 lines.append('完整四指标与全部来源：summary.csv、per_source_metrics.csv、paired_deltas.csv、frame_metrics.csv。')
 (folder/'REPORT_zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');print(record)
if __name__=='__main__':main(*sys.argv[1:])
