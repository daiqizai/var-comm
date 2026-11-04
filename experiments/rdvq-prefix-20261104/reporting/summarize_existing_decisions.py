"""Count already frozen decisions without selecting a new action or rescoring."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json

def read_csv(p):return list(csv.DictReader(Path(p).open()))
def csv_write(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--output',required=True);r=a.parse_args()
    L=Path(r.root);O=Path(r.output);O.mkdir(exist_ok=False)
    paths=[L/'p1_source_analysis_v1/test_source_results.csv',L/'p1_control_analysis_v1/test_expected_source_metrics.csv']
    source=read_csv(paths[0]);groups=defaultdict(list)
    for v in source:groups[(int(v['budget_bits']),v['policy'])].append(v)
    rows=[]
    for (budget,policy),vv in sorted(groups.items()):
        assert len(vv)==500
        rows.append(dict(budget_bits=budget,policy=policy,sources=500,null_count=sum(int(v['slices'])==0 for v in vv),full_count=sum(int(v['slices'])==28 for v in vv),
                         mean_packet_bits=sum(int(v['packet_bits']) for v in vv)/500,mean_slices=sum(int(v['slices']) for v in vv)/500))
    csv_write(O/'source_decision_counts.csv',rows)
    control=read_csv(paths[1]);by=defaultdict(dict)
    for v in control:by[(v['N'],v['snr_db'],v['source'])][v['policy']]=v
    totals=defaultdict(list);actions=Counter()
    for (N,snr,sid),v in by.items():
        c0=v['C0_calibrated_SNR_only'];c1=v['C1_source_informed_reference'];totals[(N,snr)].append((c0,c1))
        actions[(N,snr,c1['mcs'],c1['target'])]+=1
    differences=[]
    for (N,snr),vv in sorted(totals.items()):
        assert len(vv)==500
        differences.append(dict(N=N,snr_db=snr,sources=500,changed_action=sum((a['mcs'],a['target'])!=(b['mcs'],b['target']) for a,b in vv),
              changed_received_slice=sum(a['slices']!=b['slices'] for a,b in vv),changed_probability=sum(a['probability']!=b['probability'] for a,b in vv)))
    csv_write(O/'control_decision_differences.csv',differences)
    csv_write(O/'control_C1_action_counts.csv',[dict(N=N,snr_db=snr,mcs=m,target=t,sources=n) for (N,snr,m,t),n in sorted(actions.items())])
    ident=dict(inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},scope='Counts of existing registered decisions; no new selection, bootstrap, metrics, model calls or PHY')
    (O/'identity.json').write_text(json.dumps(ident,indent=2));print(json.dumps(dict(source_384=[v for v in rows if v['budget_bits']==384],control_differences=differences)))

if __name__=='__main__':main()
