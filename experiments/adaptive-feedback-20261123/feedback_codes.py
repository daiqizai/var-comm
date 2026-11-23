"""Quantized causal feedback. The receiver encodes measured CQIs; the transmitter sees only integers."""
import numpy as np


def lloyd_levels(samples, bits):
    levels = np.quantile(samples, (np.arange(2**bits) + 0.5) / 2**bits)
    for _ in range(100):
        codes = np.searchsorted((levels[1:] + levels[:-1]) / 2, samples)
        counts = np.bincount(codes, minlength=len(levels))
        sums = np.bincount(codes, weights=samples, minlength=len(levels))
        nxt = np.divide(sums, counts, out=levels.copy(), where=counts > 0)
        if np.max(abs(nxt - levels)) < 1e-6:
            break
        levels = nxt
    return levels.tolist()


def advance(value, bit, step):
    return min(24., max(-5., value + (step if bit else -step)))


def encode(cqi, scheme, books, init, step):
    """Produce the report payload at fixed scheduled instants, including warmup."""
    bits, period = scheme
    obs = np.asarray(cqi)[::period]
    if not bits:
        return np.empty(0, dtype=np.uint8)
    if step is not None:
        prev, codes = init, []
        for sample in obs:
            code = int(sample >= prev)
            codes.append(code)
            prev = advance(prev, code, step)
        return np.asarray(codes, dtype=np.uint8)
    if bits == 5:
        return np.rint((np.clip(obs, -5., 19.8) + 5.) / .8).clip(0, 31).astype(np.uint8)
    levels = np.array(books[str(bits)])
    return np.searchsorted((levels[1:] + levels[:-1]) / 2, obs).astype(np.uint8)


def decode(codes, bits, books, init, step):
    if not bits:
        return np.empty(0)
    if np.any(codes >= 2**bits):
        raise ValueError("Invalid feedback symbol")
    if step is None:
        return np.array(books[str(bits)])[codes]
    prev, values = init, []
    for code in codes:
        prev = advance(prev, int(code), step)
        values.append(prev)
    return np.array(values)


def delivered_features(values, slots, period, delay):
    last = (slots - delay) // period
    idx = last[:, None] - np.arange(4)
    if idx.min() < 0 or np.any(last * period + delay > slots):
        raise ValueError("Unavailable feedback history")
    return values[idx], (slots - delay) % period


def regression(features, target, penalty):
    mu, sd = features.mean(0), features.std(0).clip(1e-8)
    z = (features - mu) / sd
    ym = target.mean()
    beta = np.linalg.solve(z.T @ z / len(z) + penalty * np.eye(z.shape[1]),
                           z.T @ (target - ym) / len(z))
    return dict(mu=mu.tolist(), sd=sd.tolist(), beta=beta.tolist(), intercept=float(ym))


def predict(codes, slots, spec, mode, frozen):
    bits, period = spec["modes"][mode]
    cfg = frozen["models"][mode]
    if not bits:
        return np.full(len(slots), frozen["initial_db"])
    values = decode(codes, bits, frozen["books"], frozen["initial_db"], cfg["step"])
    x, phase = delivered_features(values, slots, period, spec["conditions"]["delay"])
    result = np.empty(len(slots))
    for i in np.unique(phase):
        model = cfg["by_phase"][str(i)]
        mask = phase == i
        result[mask] = (x[mask] - model["mu"]) / model["sd"] @ model["beta"] + model["intercept"]
    return result.clip(-5, 24)
