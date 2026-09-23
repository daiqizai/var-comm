"""Persistent predecessor retirement gate, independent of an SSH job-control state."""
import fcntl,os
from pathlib import Path
from latent_enhancement.runtime import digest,verify_snapshot
from .common import ROOT,OUT,read
from .coordinator import proc_identity

def live_short_prefix():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid() or (p/'cwd').resolve()!=ROOT:continue
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if b'-m' in argv and argv.index(b'-m')+1<len(argv) and argv[argv.index(b'-m')+1].startswith(b'short_prefix.'):
                state=proc_identity(int(p.name))
                if state and state['state']!='Z':result.append(state)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return result

def acquire_predecessor_lock(path=None):
    path=Path(path) if path else ROOT/'outputs/SHORT-PREFIX-20260923/controller.lock'
    handle=path.open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);return handle

def check_retired(gate):
    if gate.get('predecessor_mode')!='RETIRED_VERIFIED':raise RuntimeError('explicit verified retirement required')
    receipt=Path(gate['retirement_receipt']).resolve()
    if not receipt.is_relative_to(OUT.resolve()) or digest(receipt)!=gate['retirement_receipt_sha256']:raise RuntimeError('retirement receipt scope/hash')
    r=read(receipt)
    if r['status']!='VERIFIED_PREDECESSOR_EXITED_WITHOUT_M6_TRAINING' or r['short_prefix_release_allowed'] is not False:raise RuntimeError('retirement/release scope')
    if r['held_controller_start_ticks']!=gate['held_controller_start_ticks'] or r['held_controller_pid']!=gate['held_controller_pid']:raise RuntimeError('predecessor identity changed')
    prior=proc_identity(gate['held_controller_pid'])
    if prior and prior['start_ticks']==gate['held_controller_start_ticks'] and prior['state']!='Z':raise RuntimeError('original predecessor is still alive')
    # A recycled PID is never signaled. A different process at that PID is not
    # evidence that the old controller returned; inspect actual root workloads.
    live=live_short_prefix()
    if live:raise RuntimeError('short-prefix work already active: '+str(live))
    verify_snapshot(r['evidence_bindings'])
    verify_snapshot(read(ROOT/'outputs/SHORT-PREFIX-20260923/queue_identity.json'))
    return r
