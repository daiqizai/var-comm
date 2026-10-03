"""Freeze the external run after the separately registered historical repair."""
import argparse
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();out=root/'outputs/EXTERNAL-COMPARISON-20261004';runtime=out/'runtime'
    sys.path.insert(0,str(runtime))
    import external_controller as c
    evaluation=c.read(runtime/'external_eval_config.json')
    probe=c.read(out/'swin_qualification_attempt2_launch.json')['cmd']
    option=lambda key:probe[probe.index('--'+key)+1]
    old=root/'outputs/HISTORICAL-METRICS-R3-20261003'
    recovery=root/'outputs/HISTORICAL-PHASE2-RECOVERY-20261004-R1'
    if not (recovery/'queue_registration.json').exists():
        raise RuntimeError('Register the reviewed historical continuation first')
    env=dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        OPENBLAS_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src'))
    config=dict(version=c.VERSION,root=str(root),output=str(out/'controller'),mode='train',poll_seconds=20,
        old=dict(output=str(old),**{key+'_sha256':c.sha(old/(key+'.json')) for key in
            ('queue_registration','controller_launch','controller_dispatch')},
            phase2_recovery=dict(output=str(recovery),queue_registration_sha256=c.sha(recovery/'queue_registration.json'))),
        training=dict(python=probe[0],script=str(runtime/'swin_train.py'),vendor=option('vendor'),
            output=option('output'),config=option('config'),image_cache=option('image-cache'),
            reference_registration=option('reference-registration'),environment=dict(env)),
        hifi=dict(python=probe[0],preflight_script=str(runtime/'hifi_preflight.py'),
            qualification_script=str(runtime/'hifi_qualification.py'),diffcom_vendor=evaluation['diffcom_vendor'],
            adm_checkpoint=evaluation['adm_checkpoint'],preflight_output=str(out/'hifi_preflight'),
            qualification_output=str(out/'hifi_qualification'),environment=dict(env)),
        evaluation=dict(script=str(runtime/'external_eval.py'),config=str(runtime/'external_eval_config.json'),
            output=evaluation['output'],reconstruction_python=probe[0],
            score_python=str(root/'outputs/UNIFIED-METRICS-20261002/environment/bin/python'),
            reconstruction_environment=dict(env),score_environment=dict(env)))
    config['bindings']={str(p):c.sha(p) for p in c.required_sources(config)}
    c.validate_config(config)
    # Reuse the already measured genuine probes only after their full checks.
    for stage in ('qualification','hifi_preflight'):
        if not c.completion(config,stage):raise RuntimeError('Missing actual GPU precondition: '+stage)
    c.register(out/'controller_config.json',config)
    print('EXTERNAL_CONTROLLER_REGISTERED',c.sha(out/'controller_config.json'),flush=True)


if __name__=='__main__':main()
