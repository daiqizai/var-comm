"""One closed-data CPU statistics run; no image/model/channel access."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import argparse,traceback,importlib.abc
import numpy as np
from analysis import merge_scores,summarize,capacity_summary,validate_bank
from contracts import check_closed,sources_match,normalize_nest,score_plan_match,baseline_summary


def load_csv(d):
    with checked(d).open(newline='')as f:return list(csv.DictReader(f))


def load_closed(c):
    role=c['role'];reg=cr(c['registration']);wait=cr(c['actual_wait']);done=cr(c['completion'])
    check_closed(wait,done,c['registration'],reg,role,gone)
    for suffix in('','_control','_initial_control'):assert not(L/(role+suffix)/'failure.json').exists()
    for d in reg['code']:checked(d)
    for d in(reg['scope'],reg['revision'],reg['frozen_policy']):checked(d)
    assert reg['revision']['sha256']=='fb80b863dad31fa37cac9cc2e8d8ee4a7d5248071b3a73fdcc74c704517fc254'
    assert reg['frozen_policy']['sha256']=='68d12e9b19006e8509b3ae5a62132e73044b88ad29ca041f96b436144923ae8c'
    assert reg['hostname']==os.uname().nodename
    assert c['scores']==done['scores']and c['plan']==reg['plan']
    for science in reg['science_closures']:
        sw=cr(science['actual_wait']);sd=cr(science['completion'])
        assert sw['actual_wait']is True and type(sw['returncode'])is int and sw['returncode']==0
        assert gone(sw['process'])and gone(sw['parent_process'])
        assert not any(sw.get(k,False)for k in('timed_out','terminated','killed'))and sw.get('metadata_error')is None
        assert sd['status']in('COMPLETE','COMPLETE_PHY_AWAITING_SOURCE_RX')
        summary=sd.get('ledger',sd.get('packet_ledger'))
        if summary is not None:assert summary['unresolved']==0
        for suffix in('','_control','_initial_control'):assert not(L/(science['role']+suffix)/'failure.json').exists()
        if science.get('audit'):
            audit=cr(science['audit']);assert audit['status']=='PASS'and not audit['violations']
            assert gone(audit['process'])and gone(audit['parent_process'])
            assert audit['actual_wait']==science['actual_wait']and audit['source_output_manifest']==sd['outputs']
    db=checked(c['ledger']);con=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
    try:
        assert con.execute("SELECT COUNT(*)FROM calls WHERE status!='COMPLETE'").fetchone()[0]==0
        counts=[list(v)for v in con.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status')]
        assert sorted(counts)==sorted(done['counts'])
    finally:con.close()
    plan=cr(c['plan']);scores=load_csv(c['scores']);score_plan_match(scores,plan,reg['metric_kinds'][role])
    assert len(scores)==done['rows']
    return scores,plan,reg,done


def main(role,path,expected):
    assert sha(path)==expected;r=read(path)
    assert r['schema']=='C_PREDCLASS_FORMAL_STATISTICS_V1'and role=='c_predclass_formal_statistics'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.uname().nodename==r['hostname']
    for d in r['code']+r['inputs']:checked(d)
    out=L/role;dest=O/'c_predclass_formal_delivery_v1';assert not dest.exists()
    assert r['destination']==str(dest)and r['no_new_bootstrap_indices']is True
    assert not out.exists();out.mkdir();beg=time.monotonic()
    save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration=pin(path)))
    forbidden={'torch','torchvision','tensorflow','transformers','diffusers','timm'}
    class NoModels(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0]in forbidden:raise RuntimeError('No model imports in CPU statistics: '+fullname)
    sys.meta_path.insert(0,NoModels())
    try:
        # All four actual metric closures must pass before any statistic.
        closed={};loaded={}
        for pop in('common500','kodak24'):
            item=r['populations'][pop]
            closed[pop]={k:load_closed(item['metrics'][k])for k in('three','clip')}
        for pop in('common500','kodak24'):
            item=r['populations'][pop];three,pt,rt,dt=closed[pop]['three'];clip,pc,rc,dc=closed[pop]['clip']
            assert rt['frozen_policy']==rc['frozen_policy']==r['frozen_policy']
            sources,proof=sources_match((pt,pc),pop,cr(item['source_identity']),load_csv(item['reference_frames']))
            if pop=='common500':assert cr(item['reference_completion'])['frame_scores']==item['reference_frames']
            nest=normalize_nest(pop,sources,[load_csv(d)for d in item['nest_source_tables']])
            oldsummary=baseline_summary(pop,load_csv(item['baseline_summary']))
            manifest=cr(item['bank_manifest'])
            bank_d=manifest['bootstrap']['indices']if pop=='common500'else manifest['bank']
            assert bank_d==item['bank'];bank=np.load(checked(bank_d),allow_pickle=False);validate_bank(bank,pop)
            if pop=='kodak24':assert manifest['source_order']==list(range(24))and manifest['seed']==2026100901
            else:assert manifest['bootstrap']['seed']==2026100701
            frames=merge_scores(three,clip,sources,pop)
            loaded[pop]=(item,sources,proof,nest,oldsummary,bank,frames)
        dest.mkdir();ledger=Ledger(out/'ledger.sqlite',dict(source_paired_summary=2))
        population_results={}
        for pop in('common500','kodak24'):
            assert time.monotonic()-beg<r['max_seconds']
            item,sources,proof,nest,oldsummary,bank,frames=loaded[pop]
            p=dest/pop;p.mkdir()
            def calculate():
                result=summarize(frames,sources,nest,bank,pop)
                capacity,distribution=capacity_summary(cr(item['choices']),sources,pop)
                csvsave(p/'frame_metrics.csv',frames)
                for name,rows in result.items():csvsave(p/(name+'.csv'),rows)
                csvsave(p/'capacity_summary.csv',capacity);csvsave(p/'endpoint_distribution.csv',distribution)
                combined=oldsummary+[dict(**v,statistics_origin='NEW_ADA_FROM_EXISTING_BANK')for v in result['summary']]
                # csvsave requires stable field order; normalize keys without decimal conversion.
                fields=list(oldsummary[0]);combined=[{k:v[k]for k in fields}for v in combined]
                csvsave(p/'combined_summary.csv',combined);save(p/'source_identity_checks.json',proof)
                save(p/'statistics_identity.json',dict(population=pop,source_order=[v['source_id']for v in sources],
                  bank=item['bank'],bank_manifest=item['bank_manifest'],new_random_draws=0,source_count=len(sources),
                  per_source_noise_count=3,unit='source image after three-noise mean',interval='pointwise percentile95; method=linear',
                  pairing='NeST source mean minus Ada source mean BEFORE fixed bank indexing; cross-method noise may differ',
                  old_baseline_statistics_recomputed=False,all_failure_denominators_retained=True,
                  actual_transmitted_prefix_is_distinct_from_accepted_prefix=True,
                  paper32_subset='Capacity diagnostic only, not separately scored or selected'))
                return dict(population=pop,logical_frames=len(frames),new_mean_intervals=len(result['summary']),
                  new_paired_intervals=len(result['paired_differences']),combined_rows=len(combined),outputs=[pin(q)for q in sorted(p.iterdir())])
            population_results[pop]=ledger.call('source_paired_summary',pop,calculate)
        assert ledger.summary()['unresolved']==0
        for d in r['inputs']+r['code']:checked(d)
        report=dict(status='C_FORMAL_SCORES_AND_SOURCE_PAIRED_STATISTICS_COMPLETE',registration=pin(path),
          frozen_policy=r['frozen_policy'],populations=population_results,ledger=ledger.summary(),new_random_draws=0,
          new_model_calls=0,new_PHY=0,new_reconstruction=0,elapsed_seconds=time.monotonic()-beg)
        assert not any(x.split('.')[0]in forbidden for x in sys.modules)
        save(dest/'completion.json',report);save(out/'completion.json',dict(status='COMPLETE',delivery=pin(dest/'completion.json'),registration_sha256=expected,ledger=ledger.summary()))
        print(json.dumps(dict(delivery=pin(dest/'completion.json'),logical_frames=4716,new_mean_intervals=24,new_paired_intervals=24)))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True)
    a=p.parse_args();main(a.role,a.registration,a.expected_registration)
