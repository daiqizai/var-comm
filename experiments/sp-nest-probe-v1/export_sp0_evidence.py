"""Materialize Pilot50 SP0 tables with missing evidence explicit. No model nor PHY calls."""
import argparse, csv, hashlib, json
from pathlib import Path
from audit_pilot import LE, rows, writecsv
from profiles import OFF, QUARTER, count


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    roster=list(csv.DictReader((a.out/'pilot50_asset_inventory.csv').open(encoding='utf8',newline='')))
    if len(roster)!=50 or [int(r['pilot_slot']) for r in roster]!=list(range(50)): raise ValueError('unexpected inventory')
    audit=json.loads((a.out/'pilot50_audit.json').read_bytes()); ident=audit['old_clean_model_identity']
    lens={(int(r['pilot_slot']),int(r['j']),int(r['m']),int(r['K'])):r for r in rows(a.repo,LE+'source_lengths.csv')}
    clean={(int(r['pilot_slot']),int(r['m']),int(r['K'])):r for r in rows(a.repo,LE+'clean_quality.csv')}
    if len(lens)!=3400 or len(clean)!=1200: raise ValueError('original archive keys')
    streams=[]; scores=[]
    for i,s in enumerate(roster):
        for j in (4,5,6,7):
            bvals={r['b_base'] for (slot,jj,_,_),r in lens.items() if slot==i and jj==j}
            if len(bvals)!=1: raise ValueError('base identity/length')
            base=bvals.pop()
            for m,k in QUARTER:
                if count((m,k))<=OFF[j]: continue
                r=lens.get((i,j,m,k))
                if (r is None)!=(m==10) or r is not None and (r['source_id']!=s['source_id'] or r['b_base']!=base): raise ValueError('unregistered split key')
                streams.append(dict(pilot_slot=i,source_id=s['source_id'],j=j,m=m,K=k,total_tokens=count((m,k)),base_bits=base,enhancement_bits=r['b_enhancement'] if r else '',endpoint_role='fallback_base' if j==4 else 'planned_base',length_evidence='LE0_LE1_archive' if r else 'MISSING_real_enhancement_length'))
        for m,k in QUARTER:
            r=clean.get((i,m,k));
            if (r is None)!=(m==10) or r is not None and (r['source_id']!=s['source_id'] or r['model_identity']!=ident): raise ValueError('unregistered score key')
            scores.append(dict(pilot_slot=i,source_id=s['source_id'],m=m,K=k,total_tokens=count((m,k)),source_pixel_sha256=s['source_pixel_sha256'],native_reconstruction_sha256=r['image_array_sha256'] if r else '',model_identity=ident,dinov2_cosine=r['dinov2_vitl14_cosine'] if r else '',psnr_db=r['psnr_db'] if r else '',dists='',dreamsim='',evidence='LE0_LE1_DINO_PSNR_only' if r else 'MISSING_all_four'))
    assert len(streams)==3600 and sum(v['enhancement_bits']!='' for v in streams)==3400 and len(scores)==1250 and sum(v['psnr_db']!='' for v in scores)==1200
    writecsv(a.out/'sp0_source_length_evidence.csv',streams); writecsv(a.out/'sp0_clean_quality_evidence.csv',scores)
    print('3600 split keys (3400 measured / 200 outstanding); 1250 clean source+endpoint keys (1200 with archived PSNR/DINO). No missing entry converted to zero.')

if __name__=='__main__':main()
