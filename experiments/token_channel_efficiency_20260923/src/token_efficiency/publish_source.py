"""Publish only verified source A records; never alter runtime evidence."""
import csv,json,shutil
from pathlib import Path
from latent_enhancement.runtime import digest,verify_snapshot,write_json
from .common import ROOT,EXP,OUT,read
from .source import freeze_targets,summarize

def copy_exact(source,destination):
    if destination.exists() and digest(destination)!=digest(source):raise RuntimeError('refusing to replace different published evidence: '+str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():shutil.copyfile(source,destination)
    return {'original_path':str(source),'published_path':str(destination),'sha256':digest(source),'bytes':source.stat().st_size}

def main():
    targets=read(EXP/'quality_targets.json');summaries={};public=ROOT/'results/token_channel_efficiency_20260923'
    for role,count in [('calibration',1000),('development',100)]:
        folder=OUT/'source'/role;done=read(folder/'completion.json');reg=read(folder/'registration.json')
        if done['sources']!=count or done['synthetic'] is not False or done['preflight_only'] is not False or done['registration_sha256']!=digest(folder/'registration.json'):raise RuntimeError('formal source completion scope')
        verify_snapshot(reg['bindings'])
        for name,sha in done['files'].items():
            if digest(folder/name)!=sha:raise RuntimeError('source output changed')
        with (folder/'source_codec_per_image.csv').open() as handle:rows=list(csv.DictReader(handle))
        if len(rows)!=25*count or len({(r['source_id'],r['method']) for r in rows})!=25*count or {r['population'] for r in rows}!={role}:raise RuntimeError('source grid')
        if role=='calibration':
            if freeze_targets(rows,digest(folder/'source_codec_per_image.csv'))!=targets:raise RuntimeError('target recomputation')
        elif reg['targets']!=targets:raise RuntimeError('development did not start with these frozen targets')
        expected={r['method']:r for r in summarize(rows)}
        with (folder/'source_codec_summary.csv').open() as handle:saved={r['method']:r for r in csv.DictReader(handle)}
        for name,record in expected.items():
            for key,value in record.items():
                actual=float(saved[name][key]) if isinstance(value,(float,int)) else saved[name][key]
                if actual!=value:raise RuntimeError('source summary differs from common source object')
        summaries[role]=saved
        dst=public/f'source_{role}'
        files={name:copy_exact(folder/name,dst/name) for name in ('source_codec_per_image.csv','source_codec_summary.csv','payload_ledger.csv','completion.json')}
        # The already published calibration lineage is preserved byte-for-byte.
        if role=='development':
            lineage={'status':'REAL_SOURCE_DEVELOPMENT_COMPLETE','population':role,'sources':count,'source_rows':done['source_rows'],'ledger_rows':done['ledger_rows'],'source_registration_sha256':digest(folder/'registration.json'),'source_code_bindings':reg['bindings'],'precision':reg['precision'],'decoder_sha256':reg['decoder_sha256'],'quality_targets_sha256':digest(EXP/'quality_targets.json'),'quality_targets_at_development_start':reg['targets'],'synthetic':False,'finite_channel_ranking':False,'new_holdout_accessed':False,'files':files,'not_published':['pixels','tokens','bitstreams','weights','tensor caches']}
            p=dst/'lineage.json'
            if p.exists() and read(p)!=lineage:raise RuntimeError('published development identity mismatch')
            if not p.exists():write_json(p,lineage)
    lines=['# Source representation and actual bitstreams: experiment A complete','',
        'Formal scope:1000 calibration and100 original development sources, kept separate. Each source has actual raw/arithmetic m6–m10 token roundtrips and20 D0/Dc representation references. The25000+2500 per-source rows and10000+1000 ledger rows are published with hashes. Two preliminary calibration probes are excluded from these formal counts. No new holdout, new training, model weights, pixels or actual token/bitstream arrays are included in this source-only publication.','',
        'These are noiseless source/interface references, with no finite-channel N or E. Arithmetic length is the actual written integer bitstream, including termination, not cross entropy. Byte-storage padding is reported separately and is not claimed as transmitted source bits. Selected payload length includes the predeclared raw fallback. Header, CRC/tail and FEC/rate-matching remain separate costs; source-bit reduction does not establish channel-use reduction.','',
        '## Actual source lengths','',
        '| Population | m | Raw bits | Arithmetic mean bits | P95 bits | Selected payload mean bits | Raw fallback | Source-bit reduction |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role in summaries:
        for m in range(6,11):
            r=summaries[role][f'codec_m{m}'];raw=float(r['raw_bits_mean']);selected=float(r['selected_payload_bits_mean'])
            lines.append(f"| {role} | {m} | {raw:.0f} | {float(r['arithmetic_bits_mean']):.3f} | {float(r['arithmetic_bits_p95']):.2f} | {selected:.3f} | {float(r['raw_fallback_fraction']):.2%} | {1-selected/raw:.2%} |")
    lines+=['','All registered levels are retained. Source-token roundtrip succeeds for every formal source/mode in both families; this is noiseless codec correctness, not wireless reliability. Full distributions (mean,median,P90/P95,min,max) and flush/storage fields are in the CSVs.','','## Decoder/representation grid','','| Population | Path | PSNR dB | LPIPS | DINO |','| --- | --- | ---: | ---: | ---: |']
    for role in summaries:
        for method in ('D0_F','D0_Fq','Dc_F','Dc_Fq'):
            r=summaries[role][method];lines.append(f"| {role} | {method} | {float(r['psnr_db_mean']):.6f} | {float(r['lpips_alex_mean']):.6f} | {float(r['dino_cosine_mean']):.6f} |")
    lines+=['','Decoder adaptation changes the interface response: on original development, Dc(F) and Dc(Fq) differ by about3.008dB and0.04656LPIPS, while D0(Fq) has higher mean DINO than Dc(Fq). These descriptive comparisons do not isolate a decoder-independent quantization loss, establish a deployed communication winner, or supply free continuous latent information to RX.','','## Prefix references: official ten-scale cumulative mapping','','| Population | Path | PSNR dB | LPIPS | DINO |','| --- | --- | ---: | ---: | ---: |']
    for role in summaries:
        for m in range(6,10):
            for decoder in ('D0','Dc'):
                for kind in ('prefix','VAR'):
                    name=f'{decoder}_{kind}_m{m}';r=summaries[role][name];lines.append(f"| {role} | {name} | {float(r['psnr_db_mean']):.6f} | {float(r['lpips_alex_mean']):.6f} | {float(r['dino_cosine_mean']):.6f} |")
    lines+=['','m10 is the complete true quantized latent Fq; no unnecessary suffix generation is performed. These direct-prefix and VAR references are not substitutes for independently trained Prefix-RX communication models.','','## Frozen common targets','','| Target | Calibration reference | PSNR minimum | LPIPS maximum |','| --- | --- | ---: | ---: |']
    for name,t in targets['targets'].items():lines.append(f"| {name} | {t['reference']} | {t['psnr_min']:.8f} | {t['lpips_max']:.8f} |")
    lines+=['','These are the registered median-based calibration targets. The exact values and calibration source-CSV SHA are in `experiments/token_channel_efficiency_20260923/quality_targets.json`. The development registration embeds the same frozen object before development loads. All targets, including unattained ones and negative resource savings, must remain in subsequent B1/B2 tables.','','## Remaining scope','','Only experiment A is complete here. P2048/P3060 have separate real-device optimizer/gradient/energy/resume acceptance, and budget training is ongoing. New digital quality grids, selected development/online timing,16QAM energy-constrained results, minimum tested N and merged short-prefix C remain incomplete. The old P4084/digital/mixed results keep their original scope; no new finite-channel ranking is inferred from this report.']
    report=ROOT/'reports/token_channel_efficiency_20260923_source_A.md';report.write_text('\n'.join(lines)+'\n')
    print('SOURCE_A_PUBLICATION_VERIFIED',{'calibration':1000,'development':100,'report':str(report)})
if __name__=='__main__':main()
