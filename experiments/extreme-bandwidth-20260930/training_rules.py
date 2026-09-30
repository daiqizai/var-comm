"""Pure CPU rules for the single authorized N512 training run."""
import math

N = 512
ARM = 'P512'
SEED = 2026093001
ORDER_SEED = 2026092301
CHANNEL_SEED = 2026092302
TRAIN_SNRS = [1, 4, 7, 13, 19]
CAL_SEEDS = [4101, 4102, 4103]
INTERVAL = 2500
MILESTONE = 20000
MAX_UPDATES = 40000
MIN_RELATIVE_GAIN = 0.002


def extension_decision(history, step):
    if step not in (20000, 30000, 40000):
        raise ValueError('Unregistered training boundary')
    by_step = {int(row['step']): float(row['utility']) for row in history}
    if len(by_step) != len(history):
        raise ValueError('Duplicate calibration candidate')
    values = [by_step[s] for s in (step-2*INTERVAL, step-INTERVAL, step)]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('Invalid full calibration utility')
    gains = [(values[i]-values[i+1])/values[i] for i in range(2)]
    gate_pass = all(g >= MIN_RELATIVE_GAIN for g in gains)
    extend = step < MAX_UPDATES and gate_pass
    reason = ('BUDGET_CAP_WHILE_IMPROVING' if gate_pass else 'BUDGET_CAP_GATE_STOP') if step == MAX_UPDATES else ('BOTH_INTERVALS_PASS' if extend else 'CALIBRATION_GATE_STOP')
    return dict(step=step, utilities=values, relative_improvements=gains,
                threshold=MIN_RELATIVE_GAIN, extend=extend,
                next_limit=step+10000 if extend else step, reason=reason,
                budget_truncated=step == MAX_UPDATES and gate_pass,
                development_used=False, convergence_claimed=False)


def select_checkpoint(history):
    if not history or any(not math.isfinite(float(r['utility'])) for r in history):
        raise ValueError('No valid completed calibration candidates')
    if len({int(r['step']) for r in history}) != len(history):
        raise ValueError('Duplicate calibration candidate')
    return min(history, key=lambda r: (float(r['utility']), int(r['step'])))
