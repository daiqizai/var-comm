"""Use freshly observed MAIN process identities for the nested PHY CPU guard.

Scientific calls and original checks are unchanged. The preceding metadata-only
delivery failure is retained; it produced no delivery data.
"""
import ast,json,os,pathlib,sys,time
import rx_boundary_finalize as original
c=original.c


def matches_closed(identity,evidence,now):
    assert evidence['stage']=='phy' and evidence['status']=='PASS'
    assert evidence['host']==identity['host']
    assert 0<=now-evidence['observed_unix']<=300
    return any(x['pid']==identity['pid'] and x['start_ticks']==identity['start_ticks'] and x['host']==identity['host'] and x['verified_gone']is True for x in evidence['processes'])


def failure_source(source):
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='failure')
    text=ast.get_source_segment(source,fn)
    assert text.count('c.gone(')==3
    return text.replace('c.gone(','cross_gone(')


def main_source(source):
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    text=ast.get_source_segment(source,fn)
    assert text.count('previous.failure()')==1
    return text.replace('previous.failure()','crosshost_failure()')


def main(expected_closure):
    q=c.read(c.O/'crosshost_finalization_qualification.json');assert q['status']=='PASS'and q['science_calls']==0
    for pin in q['sources']:c.verify(pin)
    old=c.L/'delivery_control'
    assert c.desc(old/'delivery.actual_wait.json')['sha256']=='c60db570d13d3cdbc58bf28a6ad0a30b47929f3f4ffe9f98d8a783c5c91aeb7d'
    assert c.desc(old/'delivery.stderr')['sha256']=='06ea6b523525fb59d8128f93c2111fa8f3c9863fb7c80ba34b719df1770a511d'
    assert c.read(old/'delivery.actual_wait.json')['returncode']==1
    assert c.gone(c.read(old/'delivery.actual_wait.json')['process'])and c.gone(c.read(old/'supervisor.json')['process'])
    assert not(c.O/'delivery').exists()and not(c.L/'delivery').exists()
    path=c.O/'cross_host_closure_v2.json';assert c.desc(path)['sha256']==expected_closure
    proof=c.read(path);assert proof['status']=='PASS'and proof['science_calls']==0
    evidence=next(x for x in proof['actual']if x['stage']=='phy')
    for pin in evidence['receipts']:c.verify(pin)
    def cross_gone(who):
        if who['host']==os.uname().nodename:return c.gone(who)
        return matches_closed(who,evidence,time.time())
    previous=original.previous
    ns={**vars(previous),'cross_gone':cross_gone}
    exec(compile(failure_source(pathlib.Path(previous.__file__).read_text()),'original_phy_failure::fresh_crosshost_receipt','exec'),ns)
    final_ns={**vars(original),'crosshost_failure':ns['failure']}
    exec(compile(main_source(pathlib.Path(original.__file__).read_text()),'original_delivery::fresh_crosshost_receipt','exec'),final_ns)
    final_ns['main']()
    c.put(c.O/'delivery/crosshost_closure_review.json',{'status':'PASS','science_calls':0,'prior_cpu_only_exit1':c.desc(old/'delivery.actual_wait.json'),'prior_stderr':c.desc(old/'delivery.stderr'),'fresh_cross_host_closure':c.desc(path),'foreign_gone_scope':'Only exact MAIN PHY identities in the fresh receipt; local identities still checked in /proc.'})

if __name__=='__main__':main(sys.argv[1])
