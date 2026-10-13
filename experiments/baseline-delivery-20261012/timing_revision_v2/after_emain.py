"""One wait-only successor: EMAIN actual scoring closure -> registered timing.

This is not a second scientific budget. It waits at 60-second intervals, then
exec-replaces itself with the once-only v2 owner, leaving no competing observer.
Unknown E1/timing failures stop without retry.
"""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
import timing_owner_v2 as timing

ROOT, OUT, TMP = timing.ROOT, timing.OUT, timing.TMP
E1 = ROOT / 'results/entropy_main_common500_20261013'
E1_ART = TMP / 'entropy_main_common500_20261013_v1'
WAIT = TMP / 'six_method_timing_v2_wait'
read, write, pin, sha = timing.read, timing.write, timing.pin, timing.sha


def ready_e1():
    assert not (E1_ART/'owner_failure.json').exists(), 'E1 failure retained; no automatic retry'
    assert not (OUT/'launch.json').exists(), 'Timing already attempted; no duplicate successor'
    complete, waited = E1/'scoring_completion.json', E1/'actual_wait_scoring.json'
    if not complete.exists() or not waited.exists():
        return None
    c, w = read(complete), read(waited)
    assert c['status'] == 'COMPLETE_EMAIN4500_SCORING' and c['logical_frames'] == 4500
    assert w['actual_child_waited'] is True and w['exit_code'] == w['returncode'] == 0
    assert [row['stage'] for row in c['stages']] == ['source','phy','visual','score']
    for row in c['stages']:
        assert row['actual_wait']['actual_child_waited'] is True
        assert row['actual_wait']['exit_code'] == row['actual_wait']['returncode'] == 0
        assert sha(row['completion']['path']) == row['completion']['sha256']
    for item in (c['frame_rows'], c['source_streams']):
        assert sha(item['path']) == item['sha256']
    return [pin(complete), pin(waited)]


def run():
    WAIT.mkdir(parents=True, exist_ok=True)
    lock = (WAIT/'successor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'successor_launch.json').exists(), 'Already started; inspect existing owner'
    timing.predecessor_unused()
    registration = read(OUT/'successor_registration.json')
    assert sha(__file__) == registration['source']['sha256']
    assert sha(OUT/'registration.json') == registration['timing_registration']['sha256']
    assert sha(E1/'registration.json') == registration['E1_registration']['sha256']
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                    argv=Path('/proc/self/cmdline').read_bytes().decode().split('\0')[:-1])
    write(OUT/'successor_launch.json', dict(identity=identity, started_unix=time.time(),
          registration=pin(OUT/'successor_registration.json'), wait_only=True))
    start = time.monotonic()
    try:
        while True:
            assert time.monotonic()-start < registration['max_wait_seconds'], 'Finite waiting deadline reached'
            assert not (OUT/'STOP').exists() and not (ROOT/'STOP').exists(), 'Requested STOP'
            evidence = ready_e1()
            if evidence is not None:
                # No GPU/model work is initiated by this release. The admission
                # below still rejects a live E1 science job and repository GPU job.
                if not timing.RELEASE.exists():
                    write(timing.RELEASE, dict(status='TIMING_WINDOW_RELEASED',
                        quality_jobs_stopped=True,
                        no_quality_science_until_timing_actual_exit=True,
                        evidence=evidence,
                        authority='Root coordinator explicitly authorized automatic handoff after actual E1 closure',
                        created_unix=time.time()))
                snapshot = timing.readiness()
                if snapshot['admitted']:
                    timing.verify()
                    write(OUT/'successor_handoff.json', dict(status='EXEC_HANDOFF_TO_ONCE_ONLY_TIMING_OWNER',
                          evidence=evidence, admitted_snapshot=snapshot, handed_off_unix=time.time(),
                          same_pid=os.getpid(), scientific_calls_in_waiter=0))
                    os.execv(sys.executable, [sys.executable, str(Path(timing.__file__).resolve()), 'run'])
                state = 'WAIT_SHARED_TIMING_RESOURCES'
            else:
                state = 'WAIT_E1_ACTUAL_SCORING_CLOSURE'
            write(OUT/'successor_status.json', dict(status=state, checked_unix=time.time(),
                  elapsed_wait_seconds=time.monotonic()-start, scientific_calls=0))
            time.sleep(60)
    except BaseException:
        write(OUT/'successor_failure.json', dict(status='STOPPED_NO_RETRY',
              traceback=traceback.format_exc(), stopped_unix=time.time(), scientific_calls_in_waiter=0))
        raise


if __name__ == '__main__': run()
