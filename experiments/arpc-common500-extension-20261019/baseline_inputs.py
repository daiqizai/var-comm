"""Read-only four-method join, preserving published source scores and indices."""
import math,statistics
from common import *
ALIASES={'BPG_ADAPTIVE':'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'}
def join_rows(full,nest,main,supplement):
    byindex={s['source_index']:s['source_id']for s in full}
    assert len(full)==500 and set(byindex)==set(range(500))and len(set(byindex.values()))==500
    values={}
    def add(method,snr,metric,index,value,origin):
        key=(method,int(snr),metric,int(index));assert key not in values,key
        assert method in BASELINES and int(snr)in(4,10,19)and metric in METRICS and int(index)in byindex
        assert math.isfinite(float(value))
        values[key]={'method':method,'snr_db':int(snr),'metric':metric,'source_index':int(index),'source_id':byindex[int(index)],'source_mean':float(value),'noise_count':3,'origin':origin}
    for r in nest:
        if r['population']=='common500'and r['family']=='PARTIAL':
            add('EC_VAR_PARTIAL',r['snr_db'],r['metric'],r['source_index'],r['source_mean'],'N1024_REPAIR_PARTIAL')
    for r in main:
        m=ALIASES.get(r['method'],r['method'])
        if m not in BASELINES[1:]or int(r['snr_db'])not in(4,10,19)or r['metric']not in METRICS[:4]:continue
        i=int(r['source_index']);assert r['source_id']==byindex[i]and int(r['noise_count'])==3
        add(m,r['snr_db'],r['metric'],i,r['mean'],'FINAL_MAIN')
    for r in supplement:
        m=ALIASES.get(r['method'],r['method'])
        if m not in BASELINES[1:]or int(r['snr_db'])not in(4,10,19)or r['metric']not in METRICS[4:]:continue
        assert int(r['noise_count'])==3
        add(m,r['snr_db'],r['metric'],r['source_index'],r['source_mean'],'FINAL_SUPPLEMENT')
    expected={(m,s,k,i)for m in BASELINES for s in(4,10,19)for k in METRICS for i in range(500)}
    assert set(values)==expected and len(values)==42000
    return [values[k]for k in sorted(values)]
def main():
    scope=read(O/'scope.json');pop=read(O/'population.json');out=O/'baseline_inputs_v1';assert not out.exists()
    for d in scope['baseline_inputs'].values():verify(d)
    # NeST's compact table has index only: establish the explicit source-ID join
    # from its pinned repair source table before attaching any source IDs.
    nr=R/'results/n1024_consistency_20261019_v1/repair_v1/registration.json';reg=read(nr);verify(reg['sources'])
    nsrc=read(reg['sources']['path']);assert isinstance(nsrc,list)
    nsrc={int(s['source_index']):s for s in nsrc if s['population']=='common500'}
    for s in pop['common500']:
        n=nsrc[s['source_index']]
        assert all(n[k]==s[k]for k in ('source_id','archive','preprocessing_id'))
    paths=scope['baseline_inputs'];rows=join_rows(pop['common500'],csvrows(paths['nest']['path']),csvrows(paths['main']['path']),csvrows(paths['supplement']['path']))
    points=R/'paper/manuscripts/nestcom_entropy_v6_20261019/evidence/common500_main_points.csv'
    names={'NeST-Com (entropy-coded)':'EC_VAR_PARTIAL','Latent-space JSCC':'P100000','SwinJSCC (adapted)':'Swin200000','Adaptive BPG + LDPC':'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'}
    groups={}
    for r in rows:groups.setdefault((r['method'],r['snr_db'],r['metric']),[]).append(r['source_mean'])
    comparisons=[]
    for r in csvrows(points):
        if r['method']not in names:continue
        method=names[r['method']];key=(method,int(r['snr_db']),r['metric']);mean=statistics.mean(groups[key]);scale=100 if r['metric']=='convnext_top1_source_prediction'else 1
        original=float(r['original_mean']);delta=mean-original
        comparisons.append({'method':method,'snr_db':key[1],'metric':key[2],'mean':mean,'original_mean':original,'difference':delta,'display_scale':scale,'material_difference':abs(delta)>1e-10})
    assert len(comparisons)==48 and not any(r['material_difference']for r in comparisons)
    out.mkdir();write_csv(out/'source_means.csv',rows);write_csv(out/'current_four_method_curve_check.csv',comparisons)
    put(out/'completion.json',{'status':'FOUR_METHOD_INPUTS_VERIFIED_NO_SCIENTIFIC_COMPUTE','sources':500,'source_metric_rows':42000,'figure_cells_checked':48,'material_mean_differences':0,'new_model_calls':0,'new_scores':0,'new_bootstrap_draws':0,'inputs':list(paths.values())+[desc(nr),reg['sources'],desc(points),desc(O/'population.json')],'outputs':[desc(out/n)for n in ('source_means.csv','current_four_method_curve_check.csv')]})
    print(json.dumps(desc(out/'completion.json')))
if __name__=='__main__':main()
