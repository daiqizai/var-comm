"""Original first-matrix stage order, with durable per-stage thermal-safe resumes."""
import fcntl
from latent_enhancement.runtime import write_json,verify_snapshot
from .common import OUT,read
from .C_lifecycle import SHORT
from .C_followups import CRunner

def main():
    lock=(SHORT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=read(SHORT/'queue_identity.json');verify_snapshot(identity);runner=CRunner()
    for m in (6,7,8):
        runner.run(f'C_initial_qualification_m{m}',['short_prefix.train','--group',f'm{m}','--qualification-only','--cache-from-preflight'],SHORT/f'training/m{m}_seed2026092304/qualification.json',qualification=True)
    runner.run('C_initial_execution_qualification',['short_prefix.qualify_execution'],SHORT/'qualification_execution.json',qualification=True)
    for m in (6,7,8):
        verify_snapshot(identity)
        for role in ('calibration','train'):
            runner.run(f'C_initial_cache_m{m}_{role}',['token_efficiency.C_cache','--m',str(m),'--role',role],SHORT/f'cache/N4084_m{m}_ND{ {6:1200,7:1968,8:2992}[m] }/{role}/completion.json')
        runner.run(f'C_initial_train_m{m}',['short_prefix.train','--group',f'm{m}','--until','20000'],SHORT/f'training/m{m}_seed2026092304/completion.json')
    runner.run('C_initial_pure',['short_prefix.train','--group','pure','--until','20000'],SHORT/'training/pure_seed2026092304/completion.json')
    verify_snapshot(identity)
    write_json(OUT/'C_followups/first_matrix_completion.json',{'status':'REAL_FIRST_20K_MILESTONES_COMPLETE_PENDING_CALIBRATION_DECISION','synthetic':False,'new_holdout':False,'source_bindings':identity})
if __name__=='__main__':main()
