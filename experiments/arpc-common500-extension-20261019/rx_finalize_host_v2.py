"""Read the separately checked MAIN-host qualification from the native RX host."""
import hashlib
from common import *
import rx_finalize_v1 as old

if __name__=='__main__':
    p=O/'rx_finalization_preparation_v2/cross_host_qualification_review.json';review=read(p)
    assert review['status']=='PASS' and review['all_processes_gone'] and review['host']!=os.uname().nodename
    for pin in review['evidence']:verify(pin)
    assert review['qualification']==desc(L/'rx_finalization_qualification_v1/completion.json')
    old.old_wait(L/'rx_finalization_control_v1/finalize.actual_wait.json',1)
    assert gone(read(L/'rx_finalization_control_v1/supervisor.json')['process'])
    source=E/'rx_finalize_v1.py';raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='738bf2003d682b16a154dc638dc1813aef014b5159434c6b0d2c3d799a24c39a'
    before=raw.decode();a="check_wait(q/'tests.actual_wait.json')";b="check_wait(q/'tests.actual_wait.json',live=False)"
    assert before.count(a)==1
    after=before.replace(a,b);assert after.replace(b,a)==before
    exec(compile(after,str(source)+'::cross_host_wait_verified','exec'),{'__name__':'__main__','__file__':str(source)})
