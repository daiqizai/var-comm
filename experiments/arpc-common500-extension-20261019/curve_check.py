"""Compare existing and newly unified saved plot cells; no interval computation."""
from common import *
def main():
    out=O/'curve_check_v1';assert not out.exists()
    done=read(O/'delivery_v1/completion.json')
    for d in done['outputs']:verify(d)
    before=R/'paper/manuscripts/nestcom_entropy_v6_20261019/evidence/common500_main_points.csv'
    original=read(O/'baseline_inputs_v1/completion.json');pin=next(d for d in original['inputs']if d['path']==str(before));verify(pin)
    names={'NeST-Com (entropy-coded)':'EC_VAR_PARTIAL','Latent-space JSCC':'P100000','SwinJSCC (adapted)':'Swin200000','Adaptive BPG + LDPC':'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'}
    lookup={(r['method'],int(r['snr_db']),r['metric']):r for r in csvrows(O/'delivery_v1/common500_summary.csv')}
    rows=[]
    for r in csvrows(before):
        if r['method']not in names:continue
        n=lookup[names[r['method']],int(r['snr_db']),r['metric']]
        row={'method':n['method'],'snr_db':n['snr_db'],'metric':n['metric'],'display_scale':100 if n['metric']=='convnext_top1_source_prediction'else 1}
        for k in ('mean','ci_low','ci_high'):
            row['old_'+k]=float(r['original_'+k]);row['new_'+k]=float(n[k]);row[k+'_difference']=row['new_'+k]-row['old_'+k]
        rows.append(row)
    assert len(rows)==48 and all(abs(r['mean_difference'])<=1e-10 for r in rows)
    changed=[r for r in rows if max(abs(r['ci_low_difference']),abs(r['ci_high_difference']))>1e-10]
    out.mkdir();write_csv(out/'four_method_main_curve_check.csv',rows)
    put(out/'comparison.json',{'status':'PASS_MEANS_UNCHANGED','cells':48,'mean_material_differences':0,'changed_interval_cells':len(changed),'max_abs_bound_change_by_metric':{m:max(max(abs(r['ci_low_difference']),abs(r['ci_high_difference']))for r in rows if r['metric']==m)for m in {r['metric']for r in rows}},'reason':'All four baseline source scores are unchanged. This extension applies the same already registered500-source bank to all five methods as requested; old plotted marginal intervals are retained as historical values, and any Monte Carlo interval differences are explicitly listed. No old-method inference or scoring was repeated.','inputs':[pin,desc(O/'delivery_v1/common500_summary.csv'),desc(O/'delivery_v1/statistics_protocol.json')],'output':desc(out/'four_method_main_curve_check.csv')})
    print(json.dumps(read(out/'comparison.json')))
if __name__=='__main__':main()
