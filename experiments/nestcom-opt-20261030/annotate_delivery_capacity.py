"""Read-only historical capacity supplement; no PHY, model, metric or bootstrap calls."""
from opt_common import *
from route_statistics_v2 import csvout
import csv,collections,shutil

def table(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))

def main():
 out=D/'delivery';validation=out/'validation';audit=validation/'capacity_reporting_annotation.json'
 assert not audit.exists()
 immutable=[pin(D/'final_common500_delivery'/n)for n in ['frame_metrics.csv','per_source_metrics.csv','summary.csv','paired_deltas.csv','endpoint_and_failure_summary.csv','statistics_identity.json']]
 sources=cr(read(D/'frozen_final.json')['common500_sources']);sm={v['source_id']:v for v in sources}
 old=B/'adatoken_response_20261026_v1';seven=B/'nestcom_7db_common500_20261030_v1'
 inputs=[]
 def load(p):inputs.append(pin(p));return read(p)
 choices={v['key']:v for v in load(old/'c_predclass_common500_link_preparation/choices.json')}
 for manifest in [load(old/'c_predclass_common500_link_preparation/sources.json'),load(R/'results/nestcom_7db_common500_20261030_v1/source_manifest.json')]:
  assert len(manifest)==500
  for s in manifest:
   f=sm[s['source_id']];assert s['source_index']==f['source_index']and s['archive']['sha256']==f['archive']['sha256']and s['preprocessing_id']==f['preprocessing_id']
   if 'reference_sha256'in s:assert s['reference_sha256']==f['reference_sha256']
 lengths={}
 for s in sources:
  with np.load(checked(s['ada_streams']),allow_pickle=False)as z:lengths[s['source_id']]={int(k[1:]):len(z[k])for k in z.files if k.startswith('p')}
  assert len(lengths[s['source_id']])==256
 physical={}
 for desc in load(old/'c_predclass_common500_link/frames.json'):
  f=cr(desc);v=f['event'];c=choices[f['choice_key']];t=f['transmission'];key=(1024,int(v['snr_db']),v['source_id'],int(v['noise_seed']))
  assert key not in physical and c['source_id']==v['source_id']and c['source_index']==sm[v['source_id']]['source_index']and c['snr_db']==v['snr_db']
  assert not c['explicit_null']and c['paid_class_bits']==10
  bits=c['arithmetic_bits'];cap=c['arithmetic_capacity_bits'];prefix=c['prefix']
  assert bits==lengths[v['source_id']][prefix]and cap==t['k']-39 and t['arithmetic_bits']==bits+10 and t['m']==prefix
  physical[key]=dict(arithmetic_bits=bits,class_bits=10,arithmetic_capacity_bits=cap,body_information_bits=t['k'],transmitted_prefix=prefix,record_path=desc['path'],record_sha256=desc['sha256'])
 for f in load(seven/'phy_ada/frames.json'):
  t=f['transmission'];key=(1024,7,f['source_id'],int(f['noise_seed']));assert key not in physical and not f['no_fit']
  assert f['source_index']==sm[f['source_id']]['source_index']and f['capacity_bits']==t['k']-29
  bits=int(f['source_bits'])-10;cap=int(f['capacity_bits'])-10;prefix=int(f['endpoint'][0])
  assert bits==lengths[f['source_id']][prefix]and t['arithmetic_bits']==bits+10 and t['m']==prefix
  desc=pin(seven/'phy_ada'/(f['key']+'.json'));assert read(desc['path'])==f
  physical[key]=dict(arithmetic_bits=bits,class_bits=10,arithmetic_capacity_bits=cap,body_information_bits=t['k'],transmitted_prefix=prefix,record_path=desc['path'],record_sha256=desc['sha256'])
 n512=R/'results/nestcom_n512_20261028_v1/delivery/frame_scores.csv';inputs.append(pin(n512))
 for v in table(n512):
  if v['method']!='ADA':continue
  key=(512,int(v['snr_db']),v['source_id'],int(v['noise_seed']));assert key not in physical
  s=sm[v['source_id']];assert int(v['source_index'])==s['source_index']and v['reference_sha256']==s['reference_sha256']
  bits=int(v['arithmetic_bits']);cap=int(v['arithmetic_capacity_bits']);k=int(v['k']);assert int(v['class_bits'])==10 and cap==k-39
  physical[key]=dict(arithmetic_bits=bits,class_bits=10,arithmetic_capacity_bits=cap,body_information_bits=k,transmitted_prefix=int(v['actual_m']),record_path=str(n512),record_sha256=sha(n512))
 rows=[]
 for v in table(D/'final_common500_delivery/frame_metrics.csv'):
  if v['method']!='Ada_frozen':continue
  key=(int(v['N']),int(v['snr_db']),v['source_id'],int(v['noise_seed']));x=physical[key];s=sm[v['source_id']]
  assert int(v['source_index'])==s['source_index']and v['source_pixels_sha256']==s['preprocessing_id']and v['reference_sha256']==s['reference_sha256']
  util=x['arithmetic_bits']/x['arithmetic_capacity_bits'];assert 0<=util<=1
  if v['utilization']:assert abs(util-float(v['utilization']))<1e-12
  rows.append(dict(method='Ada_frozen',N=key[0],snr_db=key[1],source_id=key[2],source_index=int(v['source_index']),noise_seed=key[3],channel_repeat=int(v['channel_repeat']),source_pixels_sha256=v['source_pixels_sha256'],reference_sha256=v['reference_sha256'],receiver_status=v['receiver_status'],**x,arithmetic_utilization=util,paid_container_utilization=(x['arithmetic_bits']+10)/(x['body_information_bits']-29)))
 assert len(rows)==10500 and len(physical)==10500
 csvout(out/'Ada_capacity_identity_supplement.csv',rows)
 groups=collections.defaultdict(list)
 for v in rows:groups[(v['N'],v['snr_db'])].append(v)
 capacities=table(out/'endpoint_and_failure_summary.csv');original_summary=pin(out/'endpoint_and_failure_summary.csv')
 shutil.copyfile(out/'endpoint_and_failure_summary.csv',out/'endpoint_and_failure_summary_before_annotation.csv')
 for v in capacities:
  if v['method']=='Ada_frozen':
   g=groups[(int(v['N']),int(v['snr_db']))];assert len(g)==1500
   u=float(np.mean([x['arithmetic_utilization']for x in g]));v['mean_source_utilization']=u
   v['capacity_denominator']='Arithmetic bits excluding paid class / (body information bits minus29 length+CRC minus10 class); not airtime efficiency'
   v['capacity_origin']='Identity-joined actual historical records; Ada_capacity_identity_supplement.csv'
   v['full_representation_transmissions']=sum(x['transmitted_prefix']==256 for x in g)
  else:
   v['capacity_origin']='Unchanged final statistics values';v['full_representation_transmissions']=''
 # Only the presentation copy is replaced; original statistics stay byte-identical.
 with(out/'endpoint_and_failure_summary.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in capacities for k in v)));w.writeheader();w.writerows(capacities)
 report_path=out/'REPORT_zh.txt';original_report=pin(report_path);text=report_path.read_text();shutil.copyfile(report_path,out/'REPORT_before_annotation_zh.txt')
 anchor='核心固定 FIT 分母汇总：'
 assert text.count(anchor)==1
 text=text.replace(anchor,'不利结果：N1024、7 dB 的拒收/灰图由原 NeST 的 6/1500 增至最终 D 的 13/1500；均保留在分母。该点 PSNR 配对区间支持改善，但 LPIPS、DreamSim 与 CLIP-I 的逐点区间均跨零，不能写成每个指标在每个 SNR 均显著改善。核心两点平均改善也不能代替此逐点结论。\n'+anchor)
 for v in capacities:
  if v['method']!='Ada_frozen':continue
  prefix=f"Ada_frozen N{v['N']} {v['snr_db']}dB: "
  matches=[line for line in text.splitlines()if line.startswith(prefix)];assert len(matches)==1
  line=matches[0];start=line.index('源码利用率');end=line.index('，状态',start)
  text=text.replace(line,line[:start]+'源码利用率'+str(v['mean_source_utilization'])+line[end:])
 old_definition='利用率为算术源码 bit /（body 信息容量−29bit），不是总空口效率。'
 assert old_definition in text
 text=text.replace(old_definition,'源码利用率的分子为独立终止算术码流位数。NeST/WHOLE 分母为 body 信息容量−29bit；Ada 分母再扣除实际付费类别的10bit，即 body 信息容量−39bit。N1024 Ada 的旧 PHY 通用字段 arithmetic_bits 含类别容器，本交付已按实际码流身份核对并减去10bit；N512 原字段本就不含类别。该元数据补充没有改动任何质量值、CI、接收结果或原统计文件，不是总空口效率。N1024、19 dB 的 Ada 已发送完整256 tokens，剩余容量不表示装入失败。')
 text+='\n交付口径补注：容量字段说明修订与逐帧历史补充见 validation/capacity_reporting_annotation.json；原统计字节保留。A 端点覆盖及条件均值、B 三生成种子的源内变异、C token CE/accuracy 与 content−control 差值均另附 CSV，不用于重新选择候选。\n'
 report_path.write_text(text,encoding='utf8')
 for desc in immutable:checked(desc)
 save(audit,dict(status='PASS_METADATA_ONLY',scientific_calls=0,quality_or_CI_changed=False,original_statistics_unchanged=immutable,original_presentation_summary=original_summary,corrected_summary=pin(out/'endpoint_and_failure_summary.csv'),original_report=original_report,updated_report=pin(report_path),supplement=pin(out/'Ada_capacity_identity_supplement.csv'),joined_logical_rows=len(rows),all_source_ID_pixel_preprocessing_and_reference_identity_verified=True,source_stream_file_hashes_and_actual_terminated_lengths_verified=True,inputs=inputs,definition='Ada arithmetic bits/(k-39); NeST/WHOLE arithmetic bits/(k-29). All failures retained.',correction='Original generic capacity label incorrectly omitted Ada paid class10; missing N1024 Ada capacity metadata now supplemented from actual historical events. Original final statistics remain immutable.',adverse_7dB_gray_original=6,adverse_7dB_gray_final=13))
 print('CAPACITY_REPORT_ANNOTATED',len(rows),[(k,float(np.mean([v['arithmetic_utilization']for v in g])))for k,g in sorted(groups.items())])

if __name__=='__main__':main()
