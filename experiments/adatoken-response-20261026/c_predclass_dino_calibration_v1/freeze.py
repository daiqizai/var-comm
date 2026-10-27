"""Freeze six-candidate cal1000 policy using original aggregate, without bootstrap."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from contracts import freeze_original,FREEZE_SHA,CHOICES
import argparse,traceback


def main(role,path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    assert r['revision']['sha256']=='fb80b863dad31fa37cac9cc2e8d8ee4a7d5248071b3a73fdcc74c704517fc254'
    assert type(r['max_seconds'])is int and 0<r['max_seconds']<=43200
    checked(r['scope']);checked(r['revision'])
    w=cr(r['score_actual_wait']);assert type(w['returncode'])is int and w['returncode']==0
    assert gone(w['process'])and gone(w['parent_process'])
    assert not any(w.get(k,False)for k in('timed_out','terminated','killed'))and w.get('metadata_error')is None
    done=cr(r['score_completion']);assert done['status']=='CAL1000_DINO_CLOSED_POLICY_NOT_SELECTED'
    assert done['source_count']==1000 and done['logical_frames']==18000 and done['ledger']['unresolved']==0
    assert done['bootstrap_calls']==0 and done['PHY_calls']==0
    out=L/role;out.mkdir(exist_ok=False);beg=time.monotonic()
    save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
    try:
        with checked(done['logical_scores']).open(newline='')as f:rows=list(csv.DictReader(f))
        plan=cr(r['pair_plan']);assert len(plan['sources'])==1000
        original=checked(r['original_aggregate']);assert r['original_aggregate']['sha256']==FREEZE_SHA
        means,summary,policies=freeze_original(original.read_bytes(),rows,plan['sources'])
        assert time.monotonic()-beg<r['max_seconds']
        dest=Path(r['output_directory']);assert dest.parent==O;dest.mkdir(exist_ok=False)
        csvsave(dest/'source_noise_means.csv',means);csvsave(dest/'candidate_summary.csv',summary)
        save(dest/'frozen_policies.json',dict(status='FROZEN_FROM_FULL_CAL1000',created_unix=time.time(),
            registration_sha256=expected,scope_revision=r['revision'],policies=policies,
            selection='Maximum DINO-L after three noises per source and equal weighting of all1000 sources; exact ties choose lexicographically smaller candidate ID.',
            source_count=1000,logical_frames=18000,predeclared_candidates={str(k):list(v)for k,v in CHOICES.items()},
            actual_length_fallback='All1..256 independently terminated prefixes; longest legal fit, paid10-bit predicted class and original length/CRC/profile.',
            generation='Original fixed CFG4/temperature1/top_k2000/top_p1, no search.',
            new_bootstrap=0,formal_evaluation_population=['common500','Kodak24'],formal_logical_cap=4716,
            no_formal_results_used=True,original_aggregate=r['original_aggregate'],score_completion=r['score_completion'],
            source_means=pin(dest/'source_noise_means.csv'),candidate_summary=pin(dest/'candidate_summary.csv')))
        save(out/'completion.json',dict(status='CALIBRATION_POLICY_FROZEN_FORMAL_NOT_RUN',
            registration_sha256=expected,policy=pin(dest/'frozen_policies.json'),elapsed_seconds=time.monotonic()-beg,
            model_calls=0,PHY_calls=0,bootstrap_calls=0))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True)
    p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
