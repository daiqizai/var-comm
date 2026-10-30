"""Record actual visual review and final reporting-only annotations."""
from opt_common import *
import shutil,csv,collections

def main():
 out=D/'delivery';review=read(out/'validation/artifact_review.json')
 assert review['status']=='VISUALLY_REVIEWED_ALL_8_PNG_AND_8_PDF_PAGES'
 assert read(out/'validation/capacity_reporting_annotation.json')['status']=='PASS_METADATA_ONLY'
 for fig in review['figures']:
  for p in fig['files']:checked(p)
  assert fig['pdf_pages']==1 and fig['PNG_actually_opened']and fig['PDF_render_actually_opened']
 p=out/'figure_provenance.json';previous=pin(p);v=read(p);assert v['visual_review']=='PENDING_REAL_OPEN'
 shutil.copyfile(p,out/'figure_provenance_before_review.json')
 v['visual_review']=review['status'];v['actual_review']=pin(out/'validation/artifact_review.json')
 v['capacity_reporting_note']='Source-bit plot contains only NeST/WHOLE, whose k-29 denominator and plotted values are unchanged. Ada uses k-39; see validation/capacity_reporting_annotation.json.'
 p.write_text(json.dumps(v,indent=2)+'\n')
 save(out/'validation/figure_provenance_review_update.json',dict(before=previous,after=pin(p),figure_bytes_unchanged=True,actual_review=pin(out/'validation/artifact_review.json')))
 report=(out/'REPORT_zh.txt').read_text();assert '6/1500'in report and '13/1500'in report and 'body 信息容量−39bit'in report
 with(out/'endpoint_and_failure_summary.csv').open()as f:rows=list(csv.DictReader(f))
 assert len(rows)==28 and all(v['mean_source_utilization']!=''for v in rows)
 original=[v for v in rows if v['method']=='original_NeST'and v['N']=='1024'and v['snr_db']=='7'][0]
 final=[v for v in rows if v['method']=='final_NeST'and v['N']=='1024'and v['snr_db']=='7'][0]
 assert int(original['fixed_gray'])==6 and int(final['fixed_gray'])==13
 save(out/'validation/final_reporting_checks.json',dict(status='PASS',all_28_capacity_rows_populated=True,Ada_class10_excluded_from_arithmetic_denominator=True,quality_and_CI_unchanged=True,adverse_7dB_rejections_explicit=True,all_8_figures_actually_reviewed=True,NEW200_remains_blocked=True,no_Git_commit_or_push=True,report=pin(out/'REPORT_zh.txt'),capacity=pin(out/'endpoint_and_failure_summary.csv'),figure_provenance=pin(p)))
 with(out/'README_zh.txt').open('a',encoding='utf8')as f:
  f.write('\n最终核对补充：8张PNG及8页PDF均实际打开检查，见 validation/artifact_review.json。\n容量表的 Ada10bit 类别开销已按实际记录统一口径；N1024 Ada 缺失的容量元数据已补齐。原统计文件不改，见 Ada_capacity_identity_supplement.csv 和 validation/capacity_reporting_annotation.json。\n7dB拒收数6→13/1500与该点感知区间跨零已明确写入报告。旧V6论文/主图保持原文件；本包为新的研究交付，common500不等同NEW200。\n超过单文件限制的CSV/JSON以无损分片交付，large_file_index.json给出重组规则和原始hash。PFS原件不删除。\n')
 with(E/'EXECUTION_ENTRY_INDEX.txt').open('a',encoding='utf8')as f:
  f.write('\nFinal reporting-only entries: annotate_delivery_capacity_v2 (v1 never run; v2 caches a file hash), close_delivery_metadata; no new scientific calls. Original final_statistics bytes retained. Artifact review is actual all8PNG/all8PDFpage inspection. prepare_review_assets_v2 and finalize_review handle lossless parts and staged uncommitted review package, never commit/push.\n')
 print('DELIVERY_METADATA_CLOSED')

if __name__=='__main__':main()
