"""CPU audit of a superseded hybrid-repeat boundary; never recreate a receipt.

The exported boundary is derived from the original full checkpoint, calibration
and logged lifecycle. Its JSON is an audit product, not an original completion.
"""
import ast
from pathlib import Path
from tools.publish_c_initial_milestone import OUT, check, read, digest, write
from tools import publish_c_repeat_initial as initial
from tools.audit_c_terminal_boundary import validate_thermal_sequence


def validate_superseded(current, regsha, until, decision):
    check(current['registration_sha256'] == regsha and
          current['state']['step'] > until and
          current['state']['last_full'] >= until and
          decision['step'] == until and decision['extend'] is True and
          decision['until'] == until + 10000 and
          decision['development_used'] is False,
          'only superseded, formally extended historical boundary')


def selected_from_checkpoint(src, state, arms, regsha):
    check(set(state['selection']) == set(arms), 'checkpoint selection arms')
    selected = {}
    for arm, item in state['selection'].items():
        check(type(item['step']) is int and 0 <= item['step'] <= state['step']
              and item['step'] % 2500 == 0, 'selected historical calibration step')
        p = src/'checkpoints'/f"step_{item['step']:05d}.pt"
        selected[arm] = dict(item, checkpoint=str(p), checkpoint_sha256=digest(p),
                            arm_key=arm, registration_sha256=regsha,
                            total_updates=item['step'], parent_updates=0)
    return selected


def terminal_log_summary(log, until, expected):
    prefix = f'full calibration {until} '
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    check(lines and lines[-1].startswith(prefix) and
          not any(x in log for x in ('Traceback', 'RuntimeError', 'FAILED_')),
          'terminal full calibration without later or unknown failure')
    check(ast.literal_eval(lines[-1][len(prefix):]) == expected,
          'logged original full calibration matches archived summary')


def audit(src, command, group, seed, regsha, until):
    check(group in ('m6', 'm8') and seed in initial.SEEDS,
          'historical audit supports original hybrid repeat engine only')
    check(command == ['short_prefix.train','--group',group,'--seed',str(seed),
                       '--until',str(until)], 'exact historical command')
    key = f'C_N4084_{group}_seed{seed}_until{until}'
    chain = OUT/'delivery_chain_v1'
    check(not (chain/'stages'/f'{key}.json').exists(), 'original stage takes precedence')
    current_path = src/'completion.json'
    dp = src/'delivery_decisions'/f'at_{until}.json'
    current, decision = read(current_path), read(dp)
    validate_superseded(current, regsha, until, decision)
    cp = src/'checkpoints'/f'step_{until:05d}.pt'
    terminal = initial.cpu_checkpoint(cp, until, regsha, initial.ARMS[group])
    selected = selected_from_checkpoint(src, terminal['state'], initial.ARMS[group], regsha)
    attempts = sorted((chain/'attempts').glob(key+'_*'))
    check(bool(attempts), 'original historical terminal attempt')
    inner = attempts[-1]
    launch = read(inner/'launch.json')
    check(launch['command'][3:] == command and
          launch['session_id'] == launch['process']['pid'], 'historical launch identity')
    def exited(record):
        p = Path('/proc')/str(record['process']['pid'])/'stat'
        if p.exists():
            check(p.read_text().split()[21] != str(record['process']['start_ticks']),
                  'original historical process must have exited')
    exited(launch)
    outers = [(p, read(p/'launch.json')) for p in sorted((chain/'attempts').glob('C_followups_*'))]
    before = [(p,x) for p,x in outers if x['time'] < launch['time']]
    check(bool(before), 'historical enclosing runner')
    outerpath, outer = before[-1]
    exited(outer)
    after = [(p,x) for p,x in outers if x['time'] > cp.stat().st_mtime]
    check(bool(after), 'resumed runner after original full checkpoint')
    resumedpath, resumed = after[0]
    stop, outerstop = read(inner/'stop_request.json'), read(outerpath/'stop_request.json')
    # This timestamp is the original full CHECKPOINT mtime, not completion mtime.
    validate_thermal_sequence(launch, stop, outer, outerstop, resumed,
                              cp.stat().st_mtime, dp.stat().st_mtime)
    cal = src/'calibration'/f'full_{until:05d}.json'
    csv = cal.with_suffix('.csv')
    receipt = read(cal)
    check(receipt['step'] == until and digest(csv) == receipt['sha256'],
          'original historical calibration receipt and CSV')
    terminal_log_summary((inner/'console.log').read_text(), until, receipt['summary'])
    check(cal.stat().st_mtime <= cp.stat().st_mtime and
          (inner/'console.log').stat().st_mtime <= cp.stat().st_mtime,
          'full calibration persisted before final checkpoint')
    following = sorted((chain/'attempts').glob(f'C_N4084_{group}_seed{seed}_until{until+10000}_*'))
    check(bool(following), 'original next extension actually launched')
    nextlaunch = read(following[0]/'launch.json')
    check(nextlaunch['command'][3:] == command[:-1]+[str(until+10000)] and
          dp.stat().st_mtime <= nextlaunch['time'], 'decision precedes correct next extension')
    files = {}
    for name,p in [('inner_launch.json',inner/'launch.json'),
                   ('inner_stop.json',inner/'stop_request.json'),
                   ('inner_console.log',inner/'console.log'),
                   ('outer_launch.json',outerpath/'launch.json'),
                   ('outer_stop.json',outerpath/'stop_request.json'),
                   ('resumed_launch.json',resumedpath/'launch.json'),
                   ('next_extension_launch.json',following[0]/'launch.json')]:
        files['thermal/'+name] = p
    evidence = dict(complete_stage=False, process_returncode=None,
        original_completion_available=False, original_completion_overwritten=True,
        derived_from_checkpoint=True, original_command=command,
        checkpoint_mtime=cp.stat().st_mtime,
        calibration_mtime=cal.stat().st_mtime, console_mtime=(inner/'console.log').stat().st_mtime,
        decision_mtime=dp.stat().st_mtime, terminal_checkpoint=terminal,
        overwriting_completion=dict(path=str(current_path),sha256=digest(current_path),
                                    observed_step=current['state']['step']),
        helper_sha256=digest(__file__),
        explanation='Historical full checkpoint and calibration independently verified after thermal receipt gap. Original completion was overwritten. Derived boundary/selected records are audit products, not original completion or scheduler receipts; exit code remains unknown.')
    derived = dict(status='INDEPENDENT_CHECKPOINT_DERIVED_BOUNDARY_NOT_ORIGINAL_COMPLETION',
                   state=terminal['state'], selected=selected,
                   registration_sha256=regsha, provenance=evidence)
    path = OUT/'monitoring'/f'{key}_checkpoint_derived_{__import__("time").time_ns()}.json'
    write(path, derived)
    files['training/checkpoint_derived_boundary.json'] = path
    return derived, evidence, files
