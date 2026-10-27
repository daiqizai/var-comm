"""New metadata boundary tests only; no old statistics rerun."""
import copy,unittest,ast,pathlib
from contracts import *


def identities(pop='kodak24'):
    n=COUNTS[pop];sources=[dict(source_index=i,source_id=pop+'/'+str(i),preprocessing_id='a'*64,reference_sha256='b'*64,archive={'path':'synthetic/'+str(i)})for i in range(n)]
    old=copy.deepcopy(sources)
    if pop=='kodak24':
        for s in old:s['pixels_chw_uint8_sha256']=s.pop('preprocessing_id')
    p=dict(population=pop,frozen_policy={'sha256':'f'*64},sources=sources)
    frames=[]
    for s in sources:
        for snr in SNRS:
            for noise in(11,12,13):
                frames.append(dict(source_index=s['source_index'],source_id=s['source_id'],snr_db=snr,noise_seed=noise,
                  method='EC_VAR_PARTIAL'if pop=='common500'else'NEST_PARTIAL',N=1024,reference_sha256='b'*64,
                  reference_array_sha256='b'*64,array_hash_convention='dtype_shape_cbytes'))
    return (p,copy.deepcopy(p)),{'common500'if pop=='common500'else'sources':old},frames


def closure():
    role='synthetic_three';d=dict(path='/synthetic/registration.json',bytes=2,sha256='c'*64)
    r=dict(population='kodak24',limits={role:dict(load_lpips=1,load_dreamsim=1,image_embedding=48,psnr=24,lpips_pair=24,cosine_pair=24)},metric_kinds={role:'three'},stages={role:dict(argv=['/synthetic/python','-B','worker.py',role])})
    w=dict(actual_wait=True,returncode=0,timed_out=False,terminated=False,killed=False,metadata_error=None,
      registration=d,argv=r['stages'][role]['argv']+['--registration',d['path'],'--expected-registration',d['sha256']],process={'id':1},parent_process={'id':2})
    done=dict(status='COMPLETE',registration_sha256=d['sha256'],population='kodak24',rows=216,
      counts=[['load_lpips','COMPLETE',1],['load_dreamsim','COMPLETE',1],['psnr','COMPLETE',24]],new_reconstruction=0,new_PHY=0)
    return w,done,d,r,role


class MetadataContracts(unittest.TestCase):
    def test_closed_requires_actual_integer_exit_and_both_gone(self):
        args=closure();check_closed(*args,lambda v:True)
        for key,value in(('actual_wait',False),('returncode',True),('returncode',1),('timed_out',True),('metadata_error','error')):
            a=list(copy.deepcopy(args));a[0][key]=value
            with self.assertRaises(AssertionError):check_closed(*a,lambda v:True)
        with self.assertRaises(AssertionError):check_closed(*args,lambda v:v['id']==1)
    def test_wait_registration_argv_and_ledger_bindings(self):
        args=closure()
        mutations=[lambda a:a[0].update(argv=['other']),lambda a:a[0]['registration'].update(sha256='d'*64),
          lambda a:a[1].update(rows=215),lambda a:a[1]['counts'].append(['unknown','COMPLETE',1]),
          lambda a:a[1]['counts'][2].__setitem__(1,'RESERVED'),lambda a:a[1]['counts'][2].__setitem__(2,25),
          lambda a:a[1]['counts'].pop(0),lambda a:a[1].update(new_PHY=1)]
        for mutate in mutations:
            a=list(copy.deepcopy(args));mutate(a)
            with self.assertRaises(AssertionError):check_closed(*a,lambda v:True)
    def test_both_population_schemas_and_historical_noise_namespace(self):
        for pop in COUNTS:
            plans,population,frames=identities(pop)
            s,proof=sources_match(plans,pop,population,frames[::-1]);self.assertEqual(len(s),COUNTS[pop]);self.assertEqual(len(proof),COUNTS[pop])
    def test_source_or_reference_mismatch_stops(self):
        for mutate in(lambda p,o,f:p[1]['sources'][0].update(source_id='wrong'),
          lambda p,o,f:o['sources'][0].update(pixels_chw_uint8_sha256='x'*64),
          lambda p,o,f:f[0].update(reference_array_sha256='c'*64),lambda p,o,f:f[0].update(array_hash_convention='raw_only'),
          lambda p,o,f:f.__setitem__(-1,copy.deepcopy(f[0]))):
            p,o,f=identities();mutate(p,o,f)
            with self.assertRaises(AssertionError):sources_match(p,'kodak24',o,f)
    def test_common_archive_identity_is_not_ignored(self):
        p,o,f=identities('common500');o['common500'][0]['archive']['path']='other'
        with self.assertRaises(AssertionError):sources_match(p,'common500',o,f)
    def test_nest_normalization_keeps_repaired_partial_and_clip_separate(self):
        p,_,_=identities('common500');s=p[0]['sources'];tables=[[],[]]
        for src in s:
            for snr in SNRS:
                for metric in METRICS[:3]:
                    tables[0].append(dict(population='common500',family='PARTIAL',source_index=src['source_index'],snr_db=snr,metric=metric,source_mean='0.12345678901234567'))
                tables[1].append(dict(method='EC_VAR_PARTIAL',source_index=src['source_index'],source_id=src['source_id'],snr_db=snr,metric='clip_image_cosine',source_mean='0.5',noise_count=3))
        tables[0].append(dict(population='common500',family='WHOLE',source_index=0,snr_db=19,metric='psnr_db',source_mean='99'))
        rows=normalize_nest('common500',s,tables);self.assertEqual(len(rows),6000);self.assertFalse(any(v['source_mean']==99 for v in rows))
        tables[1][0]['source_id']='wrong'
        with self.assertRaises(AssertionError):normalize_nest('common500',s,tables)
    def test_kodak_normalization_rejects_missing_noise_and_nonfinite(self):
        p,_,_=identities();s=p[0]['sources'];table=[dict(method='NEST_PARTIAL',source_index=x['source_index'],source_id=x['source_id'],snr_db=snr,metric=m,mean='.3',noise_count=3)for x in s for snr in SNRS for m in METRICS]
        self.assertEqual(len(normalize_nest('kodak24',s,[table])),288)
        for k,v in(('noise_count',2),('mean','nan')):
            bad=copy.deepcopy(table);bad[0][k]=v
            with self.assertRaises(AssertionError):normalize_nest('kodak24',s,[bad])
        with self.assertRaises(AssertionError):normalize_nest('kodak24',s,[table[:-1]])
    def test_saved_metric_row_must_match_exact_plan_and_reuse(self):
        frame=dict(source_index=0,source_id='a',snr_db=4,noise_seed=1,population='kodak24',method='ADA_PREDCLASS_FIXEDFRAME',received_prefix=2,wire_status='PAYLOAD_PARSED',source_status='ACCEPTED_SOURCE',accepted=True,pixel_sha256='a'*64,reuse_old=True,old_values=dict(psnr_db=1.,lpips_alex=.2,dreamsim=.3))
        row={k:str(v)for k,v in frame.items()if k not in('reuse_old','old_values')};row.update(frame['old_values'],score_origin='EXACT_FROZEN_SOURCE_PIXEL_PAIR_REUSE')
        score_plan_match([row],dict(frames=[frame]),'three')
        with self.assertRaises(AssertionError):score_plan_match([row],dict(frames=[frame]),'clip')
        for k,v in(('dreamsim',.4),('pixel_sha256','b'*64),('accepted','False')):
            bad=copy.deepcopy(row);bad[k]=v
            with self.assertRaises(AssertionError):score_plan_match([bad],dict(frames=[frame]),'three')
    def test_baseline_decimal_strings_survive_without_statistics(self):
        rows=[dict(method=m,snr_db=s,metric=k,source_count=500,direction='higher'if k in('psnr_db','clip_image_cosine')else'lower',mean='0.12345678901234567890',ci_low='0.01234567890123456789',ci_high='0.23456789012345678901')for m in BASELINE_METHODS['common500']for s in SNRS for k in METRICS]
        result=baseline_summary('common500',rows);self.assertEqual(len(result),60)
        self.assertEqual(result[0]['mean'],rows[0]['mean'])
        with self.assertRaises(AssertionError):baseline_summary('common500',rows[:-1])
        rows[0]['direction']='invalid'
        with self.assertRaises(AssertionError):baseline_summary('common500',rows)
    def test_worker_no_random_or_model_calls_and_one_ledger_gate(self):
        root=pathlib.Path(__file__).parent;tree=ast.parse((root/'worker.py').read_bytes())
        calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        self.assertFalse(any('random'in c or c.startswith(('torch.','timm.'))for c in calls))
        self.assertIn('ledger.call',calls);self.assertIn('summarize',calls)
        compile(tree,'worker.py','exec')


if __name__=='__main__':unittest.main()
