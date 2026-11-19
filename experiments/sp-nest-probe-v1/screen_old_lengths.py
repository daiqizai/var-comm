"""Source-only capacity diagnostic from old terminated streams. No new PHY or inference."""
import argparse,collections,csv,json,statistics,sys
from pathlib import Path
from audit_pilot import rows,LE,writecsv
from profiles import capacity,BR,EPHY,QUARTER

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    rec=rows(a.repo,LE+'source_lengths.csv');idx=collections.defaultdict(dict)
    for r in rec:
        j,e=int(r['j']),(int(r['m']),int(r['K'])); k=(int(r['pilot_slot']),j)
        if e in idx[k]:raise ValueError('duplicate')
        idx[k][e]=(int(r['b_base']),int(r['b_enhancement']))
    suffix=QUARTER[-2]; out=[]
    for j in (4,5,6,7):
        vs=[idx[i,j][suffix] for i in range(50)];b=[z[0] for z in vs]; en=[z[1] for z in vs]
        for br in BR:
            for qe,er in EPHY:
                cb,ce=capacity(2,br),capacity(qe,er); can=[i for i in range(50) if b[i]<=cb];ruled_out=sum(en[i]>ce for i in can)
                out.append(dict(j_actual_candidate=j,base_rate=br,enhancement_q=qe,enhancement_rate=er,base_capacity_bits=cb,enhancement_capacity_bits=ce,registered_sources=50,base_fits=len(can),previous_endpoint=f'{suffix[0]}:{suffix[1]}',previous_E_bits_median=statistics.median(en),base_fits_but_previous_E_already_exceeds=ruled_out,upper_bound_sources_fitting_both_through_m10=len(can)-ruled_out))
    writecsv(a.out,out); print(json.dumps([r for r in out if r['base_rate']=='2/3' and r['enhancement_q']==6]))
if __name__=='__main__':main()
