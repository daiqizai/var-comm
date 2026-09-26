"""Read-only acceptance of finalized artifacts after a thermal receipt gap.

Does not reconstruct a scheduler receipt or assert a missing process exit code.
"""
from pathlib import Path
from tools.publish_c_initial_milestone import ROOT, OUT, check, read, digest

def validate_artifacts(done, status, latest, decision, until, arms, regsha):
    check(done['status']=='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
          'real milestone completion required')
    check(done['registration_sha256']==regsha and
          done['state']['step']==done['state']['last_full']==until and
          done['state']['updates']=={a:until for a in arms} and
          set(done['selected'])==set(arms), 'terminal registered boundary')
    check(status['status']=='MILESTONE_COMPLETE' and status['state']==done['state'],
          'terminal status identity')
    check(latest['step']==until and latest['reason']=='full_calibration' and
          digest(latest['path'])==latest['sha256'], 'terminal checkpoint identity')
    check(decision['step']==until and decision['until']==until and
          decision['extend'] is False and decision['development_used'] is False,
          'only finalized calibration boundary may be independently captured')

def validate_thermal_sequence(launch, stop, outer, outer_stop, resumed, completed_time, decision_time):
    check(stop['process']==launch['process'] and stop['reason']=='thermal',
          'inner thermal identity')
    check(outer_stop['process']==outer['process'] and outer_stop['reason']=='thermal',
          'outer thermal identity')
    check(outer['command'][3:]==['token_efficiency.C_followups'] and
          resumed['command'][3:]==['token_efficiency.C_followups'],
          'registered outer commands')
    for record in (stop,outer_stop):
        h=record['hardware']
        check(h['hardware_thermal_slowdown'] or h['software_thermal_slowdown'] or
              h['temperature']>=86, 'observed thermal trigger')
    check(outer['time']<launch['time']<stop['hardware']['time']<=completed_time<
          resumed['time']<=decision_time and
          launch['time']<outer_stop['hardware']['time']<=completed_time,
          'thermal completion and resumed decision chronology')

def finalized_evidence(src, group, seed, until, arms, regsha):
    chain=OUT/'delivery_chain_v1'
    key=f'C_N4084_{group}_seed{seed}_until{until}'
    check(not (chain/'stages'/f'{key}.json').exists(), 'ordinary receipt takes precedence')
    paths={n:src/n for n in ['completion.json','status.json','latest.json',
                            f'delivery_decisions/at_{until:05d}.json']}
    done,status,latest,decision=[read(p) for p in paths.values()]
    validate_artifacts(done,status,latest,decision,until,arms,regsha)
    for arm in arms:
        p=src/f'selected_{arm}.json'
        check(read(p)==done['selected'][arm], 'selected file equals terminal completion')
        paths[p.name]=p
    attempts=sorted((chain/'attempts').glob(key+'_*'))
    check(bool(attempts), 'registered attempt required')
    attempt=attempts[-1]; launch=read(attempt/'launch.json')
    expected=['short_prefix.train','--group',group,'--seed',str(seed),'--until',str(until)]
    check(launch['command'][3:]==expected and launch['session_id']==launch['process']['pid'],
          'original terminal launch')
    def exited(record):
        p=Path('/proc')/str(record['process']['pid'])/'stat'
        if p.exists():
            check(p.read_text().split()[21]!=str(record['process']['start_ticks']),
                  'recorded process must have exited, including zombies')
    exited(launch)
    outers=[(p,read(p/'launch.json')) for p in sorted((chain/'attempts').glob('C_followups_*'))]
    before=[(p,x) for p,x in outers if x['time']<launch['time']]
    check(bool(before), 'registered enclosing runner')
    outerpath,outer=before[-1]; exited(outer)
    completed_time=paths['completion.json'].stat().st_mtime
    after=[(p,x) for p,x in outers if x['time']>completed_time]
    check(bool(after), 'resumed original runner required')
    resumedpath,resumed=after[0]
    stop=read(attempt/'stop_request.json'); outer_stop=read(outerpath/'stop_request.json')
    validate_thermal_sequence(launch,stop,outer,outer_stop,resumed,completed_time,
                              paths[f'delivery_decisions/at_{until:05d}.json'].stat().st_mtime)
    console=(attempt/'console.log').read_text()
    check(f'full calibration {until} ' in console and
          not any(x in console for x in ('Traceback','RuntimeError','FAILED_')),
          'terminal calibration log without failure')
    for name,p in [('inner_launch',attempt/'launch.json'),('inner_stop',attempt/'stop_request.json'),
                   ('inner_console',attempt/'console.log'),('outer_launch',outerpath/'launch.json'),
                   ('outer_stop',outerpath/'stop_request.json'),
                   ('resumed_launch',resumedpath/'launch.json')]:
        paths[name+'.json' if name!='inner_console' else name+'.log']=p
    evidence=dict(status='INDEPENDENT_FINALIZED_ARTIFACT_CAPTURE_STAGE_RECEIPT_MISSING',
                  scheduler_receipt_present=False, process_returncode=None,
                  explanation='Original nested runner stopped during terminal calibration; completion persisted before enclosing runner exited. Resumed lifecycle finalized from completion and calibration. No scheduler receipt or exit code is reconstructed.',
                  completion_mtime=completed_time, original_command=expected,
                  sources={n:dict(path=str(p),sha256=digest(p)) for n,p in paths.items()})
    files={'training/completion_snapshot.json':src/'completion.json'}
    files.update({'terminal_evidence/'+n.replace('/','_'):p for n,p in paths.items()})
    return done,evidence,files
