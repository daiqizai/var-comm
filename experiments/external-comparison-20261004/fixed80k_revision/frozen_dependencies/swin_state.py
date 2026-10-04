"""Small CPU helpers for immutable registration and deterministic source order."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def register(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError('Existing Swin registration differs; use a new versioned run')
    if not path.exists():
        write(path, value)


def restore_training_log(path, durable_step):
    """Preserve speculative rows separately; display only checkpointed updates."""
    path = Path(path)
    if not path.exists(): return
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    steps = [r['step'] for r in rows]
    if steps != sorted(set(steps)):
        raise RuntimeError('Training log has duplicate or reordered step identities')
    keep = [r for r in rows if r['step'] <= durable_step]
    discarded = [r for r in rows if r['step'] > durable_step]
    if not discarded: return
    receipt = dict(durable_step=durable_step, rolled_back_rows=discarded,
        status='UNCOMMITTED_UPDATES_ROLLED_BACK', scientific_results=False)
    register(path.with_name('rollback_'+identity(receipt)[:16]+'.json'), receipt)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        for row in keep: stream.write(json.dumps(row)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


class SourceOrder:
    def __init__(self, count, seed):
        self.count, self.rng = int(count), np.random.default_rng(seed)
        self.order = self.rng.permutation(self.count)
        self.position, self.epoch = 0, 0

    def next(self, count):
        result = []
        while len(result) < count:
            if self.position == self.count:
                self.order = self.rng.permutation(self.count)
                self.position = 0
                self.epoch += 1
            take = min(count-len(result), self.count-self.position)
            result.extend(self.order[self.position:self.position+take].tolist())
            self.position += take
        return result

    def state_dict(self):
        return dict(count=self.count, order=self.order.tolist(), position=self.position,
            epoch=self.epoch, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        if state['count'] != self.count or sorted(state['order']) != list(range(self.count)):
            raise RuntimeError('Invalid resumed source order')
        if not 0 <= state['position'] <= self.count or state['epoch'] < 0:
            raise RuntimeError('Invalid resumed source position')
        self.order = np.asarray(state['order'], dtype=np.int64)
        self.position, self.epoch = state['position'], state['epoch']
        self.rng.bit_generator.state = state['rng']

