"""Source-first three-noise calibration; DINO-only selection, no bootstrap."""
import argparse,csv,json,pathlib,statistics,time
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine')
def aggregate(rows,decisions,population):
    groups={};seen=set()
    for row in rows:
        sid=row['source_id'];snr=int(row['snr_db']);cid=row['candidate_id'];noise=int(row['noise_seed'])
        assert sid in population and cid in decisions[str(snr)]['proposed_candidates'] and noise in [4101,4102,4103]
        key=(sid,snr,cid,noise);assert key not in seen;seen.add(key)
        groups.setdefault((sid,snr,cid),[]).append(row)
    means=[];summaries=[];policies={}
    for snr in [4,10,19]:
        for cid in decisions[str(snr)]['proposed_candidates']:
            current=[]
            for sid in sorted(population):
                items=groups[sid,snr,cid];assert {int(r['noise_seed']) for r in items}=={4101,4102,4103} and len(items)==3
                value={'source_id':sid,'snr_db':snr,'candidate_id':cid,'noise_count':3,**{m:statistics.mean(float(r[m]) for r in items) for m in METRICS}}
                means.append(value);current.append(value)
            summaries.append({'snr_db':snr,'candidate_id':cid,'source_count':len(population),**{m:statistics.mean(r[m] for r in current) for m in METRICS}})
        eligible=[r for r in summaries if r['snr_db']==snr]
        policies[str(snr)]=sorted(eligible,key=lambda r:(-r['dinov2_vitl14_cosine'],r['candidate_id']))[0]['candidate_id']
    assert len(seen)==len(population)*18
    return means,summaries,policies
def main():
    import control as c
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True);a=p.parse_args();assert c.desc(c.REG)['sha256']==a.expected_registration
    c.success(c.RUN/'score.actual_wait.json');reg=json.loads(c.REG.read_text());done=json.loads((c.OUT/'completion.json').read_text());assert done['unresolved']==0
    for d in done['outputs']:c.verify(d)
    with (c.OUT/'logical_scores.csv').open() as f:rows=list(csv.DictReader(f))
    pop={s['source_id'] for s in json.loads(pathlib.Path(reg['pair_plan']['path']).read_text())['sources']};assert len(pop)==1000
    decisions=json.loads(pathlib.Path(reg['nomination']['path']).read_text())['decisions'];means,summaries,policies=aggregate(rows,decisions,pop)
    for name,data in [('source_noise_means.csv',means),('candidate_summary.csv',summaries)]:
        with (c.OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    c.put(c.OUT/'frozen_policies.json',{'status':'FROZEN_FROM_FULL_CAL1000','created_unix':time.time(),'registration_sha256':a.expected_registration,
      'policies':policies,'selection':'Maximize DINO-L mean after each source averages three noise realizations; exact ties choose lexicographically smaller candidate ID. PSNR/LPIPS do not select.',
      'sources':1000,'logical_frames':18000,'new_bootstrap':0,'actual_length_fallback':True,'evaluation_population':'external_common100_v1','A3_logical_cap':900,
      'outputs':[c.desc(c.OUT/n) for n in ['source_noise_means.csv','candidate_summary.csv','logical_scores.csv','completion.json']]})
if __name__=='__main__':main()
