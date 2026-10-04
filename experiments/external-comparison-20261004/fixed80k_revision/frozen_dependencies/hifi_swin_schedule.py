"""Registered mechanical CBR adaptation of the author's sampling schedule.

The primary mode replaces the original ADJSCC-C2 constant 1/48 with actual
data_N/(3*256*256). The paid header consumes resources but is not an image
measurement, so it is excluded from data CBR. There is no quality-based search.
"""
import math
import numpy as np


SCHEDULE_MODES = ("actual_data_cbr", "author_fixed_1_over_48")


def sampling_schedule(log_snrs, snr_db, data_n, *, mode="actual_data_cbr",
                      multiplier=1.0, num_train_timesteps=1000, iter_num=1000,
                      skip_type="uniform", adaptive=True):
    """Compute t_start/seq without modifying an author module or tensor table."""
    if mode not in SCHEDULE_MODES:
        raise ValueError("unregistered HiFi schedule mode")
    if type(data_n) is not int or data_n <= 0:
        raise ValueError("data_n must count actual positive complex data symbols")
    if not math.isfinite(snr_db) or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("invalid SNR or author sampling multiplier")
    levels = np.asarray(log_snrs)
    if levels.shape != (num_train_timesteps,) or not np.isfinite(levels).all():
        raise ValueError("log-SNR table differs from the diffusion time grid")
    if not np.all(levels[:-1] >= levels[1:]):
        raise ValueError("author log-SNR table must descend")
    if not 0 < iter_num <= num_train_timesteps:
        raise ValueError("invalid number of author sampling iterations")
    actual_cbr = data_n / (3 * 256 * 256)
    effective_cbr = actual_cbr if mode == "actual_data_cbr" else 1 / 48
    linear_snr = 10 ** (snr_db / 10)
    # Algebraic substitution into the author formula. config.N remains a
    # dimensionless sampling multiplier, never a communication symbol count.
    dsnr = 5 * effective_cbr * math.log10(1 + linear_snr)
    if adaptive:
        t_start = int((num_train_timesteps - np.searchsorted(levels[::-1], dsnr)) * multiplier)
    else:
        t_start = num_train_timesteps - 1
    if not 0 < t_start < num_train_timesteps:
        raise ValueError("computed t_start lies outside the author's usable time grid")
    skip = num_train_timesteps // iter_num
    if skip_type == "uniform":
        seq = [i * skip for i in range(t_start // skip)]
    elif skip_type == "quad":
        seq = [int(s) for s in np.sqrt(np.linspace(0, num_train_timesteps ** 2, t_start))]
        seq[-1] -= 1
    else:
        raise ValueError("unregistered author skip_type")
    if not seq or any(t < 0 or t >= num_train_timesteps for t in seq):
        raise ValueError("sampling sequence lies outside the author's time grid")
    return {"schedule_mode": mode, "schedule_adaptive": bool(adaptive),
            "schedule_data_N": data_n, "schedule_image_real_dimensions": 3 * 256 * 256,
            "schedule_actual_data_cbr": actual_cbr, "schedule_effective_cbr": effective_cbr,
            "schedule_dsnr": dsnr, "schedule_multiplier": multiplier,
            "schedule_formula": "DSNR=5*CBR*log10(1+10^(SNR_dB/10))",
            "schedule_selection": "registered_mechanical_resource_substitution_no_quality_search",
            "t_start": t_start, "seq": seq[::-1]}


def apply_registered_schedule(schedule, config, data_n, mode="actual_data_cbr"):
    """Reuse author diffusion tables; replace only the CBR-derived start/sequence."""
    result = sampling_schedule(schedule.log_SNRs.detach().cpu().numpy(), float(config.CSNR),
        data_n, mode=mode, multiplier=float(config.N),
        num_train_timesteps=int(schedule.num_train_timesteps), iter_num=int(schedule.iter_num),
        skip_type=config.skip_type, adaptive=bool(config.CSNR_adapt_t_start))
    schedule.t_start = result["t_start"]
    schedule.seq = result["seq"]
    return {key: value for key, value in result.items() if key != "seq"}
