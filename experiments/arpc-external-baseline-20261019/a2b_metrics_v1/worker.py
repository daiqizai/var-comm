"""Original qualified metric setup and primitives, new closed-pair iteration only."""
import hashlib,pathlib,sys
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
def adapt(data):
    original=data.decode();marker="    states=json.loads(pathlib.Path(reg['clean_states']['path']).read_text())['states'];assert len(states)==700"
    assert original.count(marker)==1
    prefix=original.split(marker)[0]
    edits=[("RUN=L/'a2_metrics_v1';O=R/'results/arpc_external_baseline_20261019_v1/a2_metrics_v1'","RUN=L/'a2b_metrics_v1';O=R/'results/arpc_external_baseline_20261019_v1/a2b_metrics_v1'"),
      ("'reference_dino':100,'image_scores':800,'lpips_pair':800,'lpips_backbone_forward':1600,'reconstruction_dino':800", "'reference_dino':900,'image_scores':7200,'lpips_pair':7200,'lpips_backbone_forward':14400,'reconstruction_dino':7200"),
      ('ARPC fixed cal100 screening only; DINO-L selects, PSNR/LPIPS diagnose; no evaluation sources.', 'ARPC original cal1000 calibration only; DINO-L selects, PSNR/LPIPS diagnose; no evaluation sources.')]
    text=prefix
    for old,new in edits:
        assert text.count(old)==1;text=text.replace(old,new,1)
    inverse=text
    for old,new in reversed(edits):assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==prefix
    return text+"    from pairs import run\n    run(locals())\nif __name__=='__main__':main()\n"
def main():
    import json
    reg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--registration')+1]).read_text())
    path=E/'a2_metrics_v1/score.py';data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==reg['original_score']['sha256']
    exec(compile(adapt(data),str(path)+'::actual_calibration_pairs_v1','exec'),{'__name__':'__main__','__file__':str(path)})
if __name__=='__main__':main()
