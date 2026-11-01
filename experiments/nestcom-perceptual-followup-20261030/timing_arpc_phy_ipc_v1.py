"""Memory-only transport to the original, qualified ARPC CPU PHY runtime.

The model process keeps its original Python/Torch route. All transport cost is
inside transmit/receive calls and thus inside the measured wall boundaries.
"""
from followup_common import *
import atexit, pickle, socket, struct, subprocess, traceback


def send(sock, value):
    data = pickle.dumps(value, protocol=4)
    assert len(data) < 16*1024*1024
    sock.sendall(struct.pack('!I', len(data)) + data)


def recv(sock):
    def exact(n):
        chunks = []
        while n:
            data = sock.recv(n)
            if not data: raise EOFError('PHY transport closed')
            chunks.append(data); n -= len(data)
        return b''.join(chunks)
    size = struct.unpack('!I', exact(4))[0]
    assert 0 < size < 16*1024*1024
    return pickle.loads(exact(size))


class MemoryCalls:
    def __init__(self): self.counts = {}
    def call(self, kind, key, fn):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return fn()


def service(fd, manifest, affinity):
    sock = socket.socket(fileno=int(fd)); sock.settimeout(600)
    began = time.time(); calls = MemoryCalls(); tx_count = rx_count = 0
    try:
        os.sched_setaffinity(0, set(map(int, affinity.split(','))))
        assert len(os.sched_getaffinity(0)) == 2
        from timing_dependency_v4 import activate
        activate(json.loads(manifest))
        import torch
        torch.set_num_threads(6); torch.set_num_interop_threads(2)
        sys.path.insert(0, str(R/'experiments/five-method-timing-20261020'))
        import native
        native.paths()
        module = native.load(R/'experiments/arpc-external-baseline-20261019/phy_v1/link.py', '_e5_arpc_original_phy_ipc')
        cls = native.replace_function(module, 'Runtime', [
            ('t.set_num_threads(2);t.set_num_interop_threads(1)',
             'assert t.get_num_threads()==6 and t.get_num_interop_threads()==2')])
        rt = cls()
        rt.header.s.ROOT = B/'baseline_strength_20261011/native_build'
        rt.header.s.load_native()
        assert rt.backend.identity['torch'] == '2.11.0+cu128'
        send(sock, dict(ok=True, identity=dict(pid=ident(os.getpid()),
             torch=torch.__version__, cuda=torch.version.cuda,
             backend=rt.backend.identity, catalogue=rt.catalogue,
             affinity=sorted(os.sched_getaffinity(0)), threads=6, interop=2,
             CPU_only=True, model_calls=0, memory_transport=True)))
        while True:
            request = recv(sock); op, args = request['op'], request['args']
            if op == 'close':
                send(sock, dict(ok=True, result=dict(TX=tx_count, RX=rx_count,
                     packet_calls=calls.counts, CPU_process_wall_seconds=time.time()-began)))
                break
            if op == 'transmit':
                assert len(args) == 3
                result = rt.transmit(*args); tx_count += 1
                counts = {}
            elif op == 'receive':
                assert len(args) == 4  # observed symbols, SNR, public counter, ledger key
                before = dict(calls.counts)
                result = rt.receive(args[0], args[1], args[2], calls, args[3]); rx_count += 1
                counts = {k:v-before.get(k,0) for k,v in calls.counts.items()}
            else: raise ValueError('Unregistered PHY operation: '+op)
            send(sock, dict(ok=True, result=result, counts=counts))
    except EOFError:
        # Parent exit closes the socket; no orphan service or automatic retry.
        pass
    except BaseException:
        try: send(sock, dict(ok=False, traceback=traceback.format_exc()))
        except BaseException: pass
        raise
    finally: sock.close()


class RuntimeProxy:
    def __init__(self, manifest, out):
        self.out = Path(out); self.closed = False
        parent, child = socket.socketpair(); parent.settimeout(180)
        affinity = sorted(os.sched_getaffinity(0)); assert len(affinity) == 2
        self.log = (self.out/'phy_service.log').open('xb')
        self.process = subprocess.Popen([
            str(RT/'envs/frozen_ldpc_py311/bin/python'), '-B', __file__,
            '--service', str(child.fileno()), json.dumps(manifest), ','.join(map(str,affinity))],
            pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, cwd=R,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        child.close(); self.sock = parent
        self.process_identity = ident(self.process.pid)
        atexit.register(self.close)
        answer = recv(self.sock)
        assert answer['ok'], answer.get('traceback')
        self.identity = answer['identity']; self.catalogue = self.identity['catalogue']
        assert self.identity['pid']['pid'] == self.process.pid
        assert self.identity['affinity'] == affinity
        self.profiles = {str(p['profile_id']):p for p in self.catalogue['profiles']}
        save(self.out/'phy_service_identity.json', self.identity)

    def profile(self, endpoint, q, rate):
        rows = [p for p in self.profiles.values() if (p['endpoint'],p['q'],p['nominal_rate']) == (endpoint,q,rate)]
        assert len(rows) == 1
        return rows[0]

    def rpc(self, op, args):
        assert not self.closed and self.process.poll() is None
        send(self.sock, dict(op=op,args=args)); answer = recv(self.sock)
        assert answer['ok'], answer.get('traceback')
        return answer

    def transmit(self, payload, profile_id, counter):
        return self.rpc('transmit', [payload,profile_id,counter])['result']

    def receive(self, observed, snr, counter, calls, event):
        answer = self.rpc('receive', [observed,snr,counter,event])
        for k,v in answer['counts'].items(): calls.counts[k] = calls.counts.get(k,0)+v
        return answer['result']

    def close(self):
        if self.closed: return
        self.closed = True; report = {}
        try:
            if self.process.poll() is None:
                send(self.sock,dict(op='close',args=[])); report=recv(self.sock)
        except BaseException:
            report = dict(close_error=traceback.format_exc())
        finally:
            self.sock.close()
            try: rc = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate(); rc=self.process.wait(timeout=10)
            self.log.close()
            save(self.out/'phy_service_actual_wait.json', dict(returncode=rc,
                 original_identity=self.process_identity, final_identity=ident(self.process.pid),
                 actual_wait=True, report=report))


if __name__ == '__main__':
    assert sys.argv[1] == '--service'
    service(*sys.argv[2:])
