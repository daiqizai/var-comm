"""Continue the merged C scope from durable calibration-selected milestones."""
import fcntl
from pathlib import Path
from latent_enhancement.runtime import digest,write_json,verify_snapshot
from .common import OUT,EXP,ROOT,read,register
from .delivery_chain import Runner
from .C_lifecycle import SHORT,next_group_action,choose_candidate

DEST=OUT/'C_followups'
SEED=2026092304
class CRunner(Runner):
    def status(self,state,**extra):write_json(DEST/'status.json',{'status':state,**extra})

def train_to_decision(runner,group,seed,N=4084,fresh=False):
    scoped=N==3060 or fresh
    folder=(SHORT/f'scoped_N{N}' if scoped else SHORT)/'training'/f'{group}_seed{seed}'
    args=['token_efficiency.C_train','--N',str(N)] if scoped else ['short_prefix.train']
    args+=['--group',group,'--seed',str(seed)]
    if not (folder/'completion.json').exists():runner.run(f'C_N{N}_{group}_seed{seed}_until20000',args+['--until','20000'],folder/'completion.json')
    while True:
        action=next_group_action(folder)
        if action['kind']=='final':return folder,action
        until=action['until'];runner.run(f'C_N{N}_{group}_seed{seed}_until{until}',args+['--until',str(until)],folder/'completion.json')

def descriptor(folder,arm,N,seed):return {'method':f'{arm}_N{N}_seed{seed}','training':str(folder),'arm':arm,'N':N,'training_seed':seed}

def main():
    DEST.mkdir(parents=True,exist_ok=True);lock=(DEST/'followups.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    readiness=read(EXP/'C_followups_ready.json');verify_snapshot(readiness['bindings']);runner=CRunner();models=[];decisions={}
    for group,arms in [('m6',['H6-V','H6-P']),('m7',['H7-V','H7-P']),('m8',['H8-V']),('pure',['P4084'])]:
        folder,decision=train_to_decision(runner,group,SEED);decisions[str(folder)]=decision
        models.extend(descriptor(folder,arm,4084,SEED) for arm in arms)
    choice=choose_candidate(DEST/'candidate.json');m=choice['m']
    # Same P3060 as supplemental B. It is never trained a second time here.
    p3060=read(OUT/f'training/P3060_seed{SEED}/finalization.json')
    register(DEST/'P3060_reuse.json',{'finalization':p3060,'sha256':digest(OUT/f'training/P3060_seed{SEED}/finalization.json'),'grid_completion_sha256':digest(OUT/'continuous_grid_v1/completion.json'),'duplicate_training':False})
    for role in ('calibration','train'):
        cache=SHORT/f'cache/N3060_m{m}_ND{1200 if m==6 else 1968}'/role
        runner.run(f'C_cache_N3060_m{m}_{role}',['token_efficiency.C_cache','--role',role,'--m',str(m),'--N','3060'],cache/'completion.json')
        if role=='calibration':runner.run(f'C_pretrain_N3060_m{m}',['token_efficiency.C_qualify','--m',str(m)],DEST/f'N3060_m{m}_acceptance.json',qualification=True)
    folder,decision=train_to_decision(runner,f'm{m}',SEED,N=3060);decisions[str(folder)]=decision
    models.extend(descriptor(folder,f'H{m}-{family}',3060,SEED) for family in ('V','P'))
    for seed in choice['additional_seeds']:
        for group,arms in [(f'm{m}',[f'H{m}-V',f'H{m}-P']),('m8',['H8-V']),('pure',['P4084'])]:
            folder,decision=train_to_decision(runner,group,seed,fresh=group=='pure');decisions[str(folder)]=decision
            models.extend(descriptor(folder,arm,4084,seed) for arm in arms)
    register(DEST/'final_calibration_decisions.json',decisions)
    spec={'run_id':'SHORT-PREFIX-20260923-selected-v1','models':models,'population':'development','new_holdout':False,'output':str(DEST/'selected_grid'),'candidate_sha256':digest(DEST/'candidate.json')}
    register(DEST/'evaluation_spec.json',spec)
    runner.run('C_selected_development',['token_efficiency.C_evaluate','--spec',str(DEST/'evaluation_spec.json')],DEST/'selected_grid/completion.json')
    runner.run('C_source_report',['token_efficiency.C_report'],ROOT/'results/token_channel_efficiency_20260923/C_v1/completion.json',gpu=False)
    verify_snapshot(readiness['bindings'])
    write_json(DEST/'completion.json',{'status':'REAL_C_GRIDS_COMPLETE_REFERENCE_AUDIT_AND_PUBLICATION_PENDING','synthetic':False,'new_holdout':False,'content_selector':False,'report_completion_sha256':digest(ROOT/'results/token_channel_efficiency_20260923/C_v1/completion.json'),'final_decisions_sha256':digest(DEST/'final_calibration_decisions.json'),'evaluation_spec_sha256':digest(DEST/'evaluation_spec.json')})
if __name__=='__main__':main()
