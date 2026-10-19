"""One-shot metadata delivery of already closed work; zero scientific calls."""
from shared import *
import fcntl

def main():
    lock=(L/'execution.lock').open('r+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(O/'delivery/completion.json')['no_further_scientific_launch']
    assert read(O/'figures_v2/actual_visual_review.json')['status']=='PASS'
    for name in ['main','leo']:
        assert read(O/'final_readiness'/('process_closure_'+name+'.json'))['status']=='PASS_ALL_RECORDED_LOCAL_IDENTITIES_GONE'
    manifest=[]
    def copy(src,dest):
        assert src.is_file() and not src.is_symlink() and src.stat().st_size<3000000
        before=pin(src);b=src.read_bytes();verify(before)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(b);stream.flush();os.fsync(stream.fileno())
        assert sha(dest)==before['sha256']
        manifest.append(dict(source=before,copy=pin(dest)))
    for c in sorted((L/'controls').iterdir()):
        assert c.is_dir()
        if c.name!='native_asset_verification':
            w=read(c/'actual_wait.json')
            assert type(w['returncode'])is int
            assert w['returncode']==(1 if c.name in ['qualification_rx','qualification_arpc'] else 0)
            assert not any(w.get(k) for k in ['timed_out','terminated','killed','metadata_error'])
        for p in sorted(c.iterdir()):
            assert p.is_file()
            copy(p,O/'execution_receipts'/c.name/p.name)
    for name in ['n2048_protocol_qualification_20261019_v1','n2048_registration_20261019_v1','n2048_rx_import_revision_qualification_20261019_v1','n2048_formal_import_entry_qualification_20261019_v1']:
        src=BASE/'baseline_strength_20261011'/name
        assert read(src/'actual_wait.json')['returncode']==0
        for p in sorted(src.iterdir()):
            assert p.is_file()
            copy(p,O/'engineering_receipts'/name/p.name)
    src=L/'resource_metadata_preparation_initial_failure_v1'
    for p in sorted(src.iterdir()):
        copy(p,O/'historical_metadata_failure'/p.name)
    for phase,method in [('qualification','nest'),('qualification','arpc_resource_v1'),('formal','nest'),('formal','arpc_resource_v1')]:
        root=L/phase/method/'rx'
        # Preserve closed lightweight native-run evidence, never image/token arrays.
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix in ['.json','.txt']:
                copy(p,O/'native_execution_receipts'/phase/method/p.name)
    for relative in ['qualification/tx/completion.json','qualification/rx_import_recovery_v1/completion.json','formal/tx/completion.json','formal/rx/completion.json','metrics_v1/bootstrap/reserved.json','metrics_v1/bootstrap/complete.json']:
        p=L/relative
        if p.exists():copy(p,O/'runtime_completion_receipts'/relative)
    put(O/'closed_receipts_manifest.json',dict(status='PASS_EXACT_BYTE_COPIES',copies=manifest,scientific_calls=0,active_arrays_copied=False))
    text='''# N2048 two-method validation: final delivery

The fixed 100-source, two-method validation is complete. The reviewed final
figure is `figures_v2/n2048_full_comparison.pdf` (editable SVG and 600 dpi PNG
beside it). `figures_v1` is retained as the first export with its recorded title
clipping; v2 changes title layout only. The earlier report's v1 figure pointer
is superseded by this final index. All scientific numbers are unchanged.

- Chinese scientific report: `delivery/REPORT_zh.md`.
- Exact source lengths, overhead and actual energy: `delivery/layout_and_payload.csv`.
- Reception counts: `delivery/link_delivery.csv`.
- Complete seven-metric scores and direct paired intervals: `metrics_v1/`.
- Qualification image review: `qualification_review.json`.
- Final visual review: `figures_v2/actual_visual_review.json`.
- Actual child waits and byte-preserved failure lineage: `execution_receipts/`.
- Explicit four-repeat authorization: `rx_import_continuation_authorization.json`.
- Exact total calls: `delivery/actual_counts.json` (3648 subframe transmissions,
  7300 header/body decoder attempts, including four approved repeated decodes).

At 4 and 10 dB both methods reject all 300 objects. At 19 dB both deliver all
300 correctly. The fixed shared 64-QAM, rate-5/6 layout fits both complete
representations, but does not provide low-SNR reliability. At 19 dB NeST-Com
has favorable paired intervals for PSNR, LPIPS, DreamSim and MS-SSIM; ARPC has
a favorable DISTS interval. DINO-L and source-prediction agreement intervals
cross zero. These are pointwise intervals, without multiplicity adjustment.

The existing N1024 comparison is preserved. No new training, TX probability
passes, tokenizer calls, bandwidth sweep or further scientific stage follows.
Large restoration backups remain deferred at the user's request. The original
completion receipt's publication-pending field records its creation time;
the final normal publication receipt is delivered separately after pushing.
'''
    (O/'README.md').open('x',encoding='utf8').write(text)
    put(O/'final_delivery_manifest.json',dict(status='SCIENTIFIC_AND_VISUAL_DELIVERY_COMPLETE',registration=pin(REG),science=pin(O/'delivery/completion.json'),final_figure=pin(O/'figures_v2/manifest.json'),actual_visual_review=pin(O/'figures_v2/actual_visual_review.json'),closed_receipts=pin(O/'closed_receipts_manifest.json'),index=pin(O/'README.md'),publication_pending=True,backup='DEFERRED',additional_science_authorized=False))
    print(json.dumps(dict(copied_files=len(manifest),status='COMPLETE')))

if __name__=='__main__':main()
