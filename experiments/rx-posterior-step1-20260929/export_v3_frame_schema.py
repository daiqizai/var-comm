"""Export the requested long-form frame schema from immutable v3 A/B objects."""
from pathlib import Path
import json,csv,hashlib,math
from rx_v3_archive import unpack,original_bytes

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/rx_posterior_step1_20260929'
DEST=BASE/'revision_v3_frame_schema'
FIELDS=['task','prior','mode','output','profile','beta','lambda','eta','snr_equiv_db','noise_seed','source_id','source_index','preprocessing_sha256','psnr','lpips','dino']
FIELDS += [f'{m}_k{k}' for m in ['acc','logp_true','entropy','path_accuracy'] for k in range(1,11)]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def collect():
    A=BASE/'revision_v3_A';B=BASE/'revision_v3_B';out=[]
    for i in range(100):
        p=A/f'source_{i:03d}.json';a=read(p)
        for row in a['rows']:
            r=dict.fromkeys(FIELDS,'')
            r.update(task='A',prior=row['method'],mode='oracle' if row['method']=='O1' else 'linear',output='fuse' if row['method']=='O1' else 'lmmse',
                     profile=row['method'],eta=row['eta'],snr_equiv_db=row['snr_equiv_db'],noise_seed=row['noise_seed'],
                     source_id=a['source_id'],source_index=i,psnr=row['psnr_db'],lpips=row['lpips_alex'],dino=row['dino_cosine'])
            r['preprocessing_sha256']=a['source_identity']['rgb_sha256']
            out.append(r)
    for ent in read(B/'source_archive_manifest.json'):
        a=unpack(read(B/ent['file']))
        assert hashlib.sha256(original_bytes(a)).hexdigest()==ent['original_sha256']
        for row in a['rows']:
            for output in (['none'] if row['mode']=='TF' else ['tok','fuse']):
                r=dict.fromkeys(FIELDS,'')
                r.update(task='B',prior=row['profile'].split('_')[0],mode=row['mode'],output=output,profile=row['profile'],
                         beta=0,eta=row['eta'],snr_equiv_db=row['snr_equiv_db'],noise_seed=row['noise_seed'],source_id=row['source_id'],
                         source_index=a['source_index'],preprocessing_sha256=row['preprocessing_sha256'])
                r['lambda']=row['lambda_']
                for k,m in enumerate(row['scales'],1):
                    for name in ['acc','logp_true','entropy','path_accuracy']:
                        v=m.get(name);r[f'{name}_k{k}']='' if v is None else v
                if output!='none':
                    q=row['token_image' if output=='tok' else 'fused_image']
                    r.update(psnr=q['psnr_db'],lpips=q['lpips_alex'],dino=q['dino_cosine'])
                out.append(r)
    assert len(out)==37200 and sum(r['task']=='A' for r in out)==4200
    assert all(math.isclose(r['eta'],10**(-r['snr_equiv_db']/20),rel_tol=1e-12) for r in out)
    assert len({(r['task'],r['source_id'],r['noise_seed'],r['snr_equiv_db'],r['mode'],r['profile'],r['output']) for r in out})==len(out)
    return out

def main():
    rows=collect()
    assert not DEST.exists();DEST.mkdir()
    for j,start in enumerate(range(0,len(rows),5000)):
        p=DEST/f'frames_{j:03d}.csv'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows[start:start+5000])
    (DEST/'README.md').write_text('# Unified v3 frame schema\n\nLossless numeric projection of the previously sealed A/B result objects. A: 4200 rows. B: 12000 TF rows plus 10500 CL decisions exported once as tok and once as fuse = 33000 rows. Total 37200 rows, not independent transmissions or noise samples. TF output=none has blank image metrics; original-token CL accuracy and path-conditioned CL accuracy are separate columns. A has no token decision metrics. O1 is oracle-only. Profiles and calibration-frozen parameters remain explicit. All B decisions use beta=0; positive-beta probability-only diagnostics remain in the original B source objects and tables.\n')
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in DEST.iterdir() if p.is_file()}
    assert all(x['bytes']<10_000_000 for x in files.values())
    (DEST/'index.json').write_text(json.dumps({'status':'EXACT_REQUESTED_FRAME_SCHEMA_FROM_SEALED_V3_RESULTS','rows':len(rows),'A_rows':4200,'B_rows':33000,
        'A_index_sha256':sha(BASE/'revision_v3_A/index.json'),'B_index_sha256':sha(BASE/'revision_v3_B/index.json'),
        'exporter_sha256':sha(Path(__file__)),'fields':FIELDS,'files':files},indent=2)+'\n')
    print('EXPORTED',len(rows))
if __name__=='__main__':main()
