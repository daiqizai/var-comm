"""Pure CPU rules for the single authorized N1024 training run."""
import math

N = 1024
ARM = 'P1024'
SEED = 2026093001
ORDER_SEED = 2026092301
CHANNEL_SEED = 2026092302
TRAIN_SNRS = [1, 4, 7, 13, 19]
CAL_SEEDS = [4101, 4102, 4103]
INTERVAL = 2500
MILESTONE = 20000
MAX_UPDATES = 40000
MIN_RELATIVE_GAIN = 0.002


def verify_matched_recipe(candidate, baseline):
    """The completed P512 recipe is fixed; only budget-derived values change."""
    unchanged = (
        'training_seed', 'initialization', 'training_snrs_db', 'snr_sampling',
        'order_seed', 'channel_seed', 'training_noise_seed', 'optimizer', 'loss',
        'scale', 'decoder_state_sha256', 'decoder_gate', 'precision',
        'train_sources', 'calibration_sources', 'calibration_snrs_db',
        'calibration_noise_seeds', 'rows_per_full_calibration',
        'full_calibration_interval', 'selection', 'first_updates',
        'extension_updates', 'maximum_updates', 'extension_decision_steps',
        'extension_rule', 'reference_execution_note', 'development_read',
        'holdout_read', 'single_model', 'reference_recipe',
        'reference_registration_sha256', 'reference_model_selected_step',
        'reference_checkpoint_sha256',
    )
    for key in unchanged:
        if candidate[key] != baseline[key]:
            raise ValueError('Matched P512 recipe changed: '+key)
    if (baseline['N'], baseline['E'], baseline['arm']) != (512, 1024, 'P512'):
        raise ValueError('Expected the completed P512 baseline recipe')
    if (candidate['N'], candidate['E'], candidate['arm']) != (N, 2*N, ARM):
        raise ValueError('Expected independent P1024 with E2048')
    for key in ('width', 'residual_blocks'):
        if candidate['architecture'][key] != baseline['architecture'][key]:
            raise ValueError('Matched architecture family changed: '+key)
    if candidate['architecture']['communication_channels'] != 8 or candidate['architecture']['parameter_count'] != 495208:
        raise ValueError('P1024 budget heads differ from the registered interface')
    if candidate['noise_namespace'] != 'VAR-CONTINUOUS-1024|source_id':
        raise ValueError('P1024 channel namespace changed')
    if candidate['training_seed'] != SEED or not candidate['P1024_authorized']:
        raise ValueError('Fresh P1024 seed or authorization changed')
    return True


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
