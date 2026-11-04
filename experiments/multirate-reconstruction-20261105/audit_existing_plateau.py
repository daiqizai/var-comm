"""Two existing-record checks only: full-token transmission and reconstruction identity.

No model, metric, bootstrap, encoder, receiver or PHY execution is performed.
"""
from pathlib import Path
import argparse,csv,hashlib,json,resource,time
import numpy as np

METRICS=('psnr_db','lpips_alex','dists','dreamsim')

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    L=Path(a.previous_root);O=Path(a.output);O.mkdir(parents=True,exist_ok=False)
    begin=time.time();cpu=resource.getrusage(resource.RUSAGE_SELF)
    for folder in ['p1_source_analysis_v1','p1_source_metrics_v1']:
        w=read(L/folder/'actual_wait.json');assert w['returncode']==0 and w['child_gone']
        assert read(L/folder/'complete.json')['passed']
    manifest=L/'p1_registration_v1/evaluation_manifest.json';sources=[s for s in read(manifest)['sources'] if s['split']=='test'];assert len(sources)==500
    rawpath=L/'p1_source_metrics_v1/scores.csv';selectedpath=L/'p1_source_analysis_v1/test_source_results.csv'
    raw={(s['source'],int(s['slices'])):s for s in rows(rawpath)}
    selected={(s['source'],int(s['budget_bits']),s['policy']):s for s in rows(selectedpath)}
    evidence=[]
    for s in sources:
        sid=s['source_id'];lengths=read(s['lengths_path']);full=[v for v in lengths if v['slices']==28];assert len(full)==1
        full=full[0];assert full['transmitted_tokens']==336
        rx=Path(s['rx_directory']);receipt=read(rx/'s28_complete.json')
        assert receipt['slices']==28 and receipt['decoded_tokens']==336 and not receipt['null']
        txdir=Path(s['lengths_path']).parent
        sent=np.load(txdir/'indices.npy',allow_pickle=False);got=np.load(rx/'s28_indices.npy',allow_pickle=False)
        assert sent.shape==got.shape==(1,336) and np.array_equal(sent,got)
        image=np.load(rx/'s28_rgb.npy',allow_pickle=False)
        ih=hashlib.sha256(image.tobytes()).hexdigest();score=raw[(sid,28)]
        assert ih==receipt['rgb_sha256']==score['reconstruction_sha256']
        assert int(score['packet_bits'])==full['packet_bits']==receipt['packet_bits']
        reference=selected[(sid,-1,'complete_representation')]
        for metric in METRICS:assert float(reference[metric])==float(score[metric])
        for budget in [3072,4096]:
            point=selected[(sid,budget,'longest_feasible')]
            assert int(point['slices'])==28 and int(point['packet_bits'])==full['packet_bits']<=budget
            for metric in METRICS:assert float(point[metric])==float(reference[metric])
        evidence.append({'source_id':sid,'transmitted_tokens':336,'received_tokens':336,'full_slices':28,'actual_packet_bits':full['packet_bits'],'rgb_sha256':ih,'packet_sha256':receipt['packet_sha256'],'budget3072_same_full_output':True,'budget4096_same_full_output':True})
    with (O/'per_source_identity.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
    summarypath=L/'p1_source_analysis_v1/summary.csv';fullmeans=[v for v in rows(summarypath) if v['policy']=='complete_representation']
    endcpu=resource.getrusage(resource.RUSAGE_SELF)
    result={'passed':True,'scope':'Two read-only existing-log/native-array identity checks','sources':500,'budgets_bits':[3072,4096],'full_token_count':336,'all_received_indices_equal_transmitted':True,'all_1000_selected_positions_use_the_exact_full_representation_array':True,'new_direct_decoder_comparison':False,'interpretation':'These two longest-feasible curve points already use the complete discrete representation. There is no missing suffix to improve by changing only suffix generation. This does not establish that model scaling or a new representation will improve quality.','direct_reconstruction_scope':'The audited full receive route reconstructs the learned codebook features and calls the same image decoder; no additional 500-source direct-decoder experiment was run.','full_means_reused':fullmeans,'inputs':[{'path':str(p),'sha256':digest(p)} for p in [manifest,rawpath,selectedpath,summarypath]],'new_calls':{'model':0,'encoder':0,'receiver':0,'metric':0,'bootstrap':0,'PHY':0,'training_updates':0},'elapsed_seconds':time.time()-begin,'cpu_process_seconds':endcpu.ru_utime+endcpu.ru_stime-cpu.ru_utime-cpu.ru_stime}
    (O/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
