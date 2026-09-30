"""Calibration-only stopping rules for the authorized single P_low arm."""
import math

TRAIN_SNRS = [-5, -2, 1, 4]
EVAL_SNRS = [-5, -2, 1, 4, 13]
CAL_SEEDS = [4101, 4102, 4103]
PARENT_STEP = 27500
INTERVAL = 2500
MILESTONE = 10000
MAX_UPDATES = 30000
MIN_RELATIVE_GAIN = 0.002


def extension_decision(history, step):
    """Use current checkpoint utilities at the last three full calibrations."""
    if step not in (10000, 20000, 30000):
        raise ValueError('extension decisions occur only at registered milestones')
    by_step = {int(row['step']): float(row['utility']) for row in history}
    if len(by_step) != len(history):
        raise ValueError('duplicate calibration step')
    values = [by_step[s] for s in (step-2*INTERVAL, step-INTERVAL, step)]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('nonpositive or nonfinite calibration utility')
    improvements = [(values[i]-values[i+1])/values[i] for i in range(2)]
    extend = step < MAX_UPDATES and all(g >= MIN_RELATIVE_GAIN for g in improvements)
    return dict(step=step, utilities=values, relative_improvements=improvements,
                threshold=MIN_RELATIVE_GAIN, extend=extend,
                next_limit=step+MILESTONE if extend else step,
                reason='MAXIMUM_REACHED' if step == MAX_UPDATES else
                       'BOTH_INTERVALS_PASS' if extend else 'CALIBRATION_GATE_STOP')


def select_checkpoint(history):
    if not history:
        raise ValueError('no complete full calibration')
    if any(not math.isfinite(float(row['utility'])) for row in history):
        raise ValueError('invalid candidate utility')
    return min(history, key=lambda row: (float(row['utility']), int(row['step'])))
