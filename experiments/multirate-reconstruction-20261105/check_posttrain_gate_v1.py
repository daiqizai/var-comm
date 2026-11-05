"""CPU-only handoff tests: reject live predecessors, failures and stop markers."""
from pathlib import Path
import json,os,time
import posttrain_chain_v1 as chain
from runtime import L,write

def main():
    root=L/'posttrain_prepare_v1/gate_scenarios';root.mkdir(exist_ok=False)
    original=(chain.L,chain.OLD,chain.STOP,chain.time.sleep)
    class WouldWait(Exception):pass
    try:
        chain.L=root;chain.OLD=root/'old';chain.OLD.mkdir();(chain.OLD/'STOP_P2').write_text('paused')
        folder=root/'controllers/prior';folder.mkdir(parents=True)
        pid=os.getpid();ticks=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
        alive=dict(pid=pid,start_ticks=ticks);gone=dict(pid=999999999,start_ticks='0')
        protocol=dict(root=str(root),deadline=time.time()+60,bindings={})
        write(folder/'owner.json',alive);write(folder/'child.json',gone)
        write(folder/'actual_wait.json',dict(returncode=0,child_gone=True))
        chain.time.sleep=lambda _:(_ for _ in ()).throw(WouldWait())
        try:chain.wait_previous('prior',protocol)
        except WouldWait:pass
        else:raise AssertionError('live predecessor owner allowed handoff')
        write(folder/'owner.json',gone);write(folder/'child.json',alive)
        try:chain.wait_previous('prior',protocol)
        except WouldWait:pass
        else:raise AssertionError('live predecessor child allowed handoff')
        write(folder/'child.json',gone);assert chain.wait_previous('prior',protocol)['returncode']==0
        write(folder/'actual_wait.json',dict(returncode=1,child_gone=True))
        try:chain.wait_previous('prior',protocol)
        except AssertionError:pass
        else:raise AssertionError('failed predecessor allowed handoff')
        (root/'STOP_MAIN').write_text('user pause')
        try:chain.boundary(protocol)
        except InterruptedError:pass
        else:raise AssertionError('user pause ignored')
        assert (root/'STOP_MAIN').read_text()=='user pause'
        (root/'STOP_MAIN').unlink()
        try:chain.boundary(dict(protocol,deadline=time.time()-1))
        except TimeoutError:pass
        else:raise AssertionError('deadline ignored')
        print(json.dumps(dict(passed=True,tests=['live owner blocked','live child blocked','actual0 and both gone accepted','actual1 blocks','user STOP retained','absolute deadline enforced'],model_calls=0,training_launches=0)))
    finally:chain.L,chain.OLD,chain.STOP,chain.time.sleep=original

if __name__=='__main__':main()
