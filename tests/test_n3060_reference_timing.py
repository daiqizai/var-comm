"""Synthetic gate tests; the deferred GPU stage performs real archived parity."""
import numpy as np
import pytest
from tools.replay_n3060_reference_timing import parity,release_ready


def test_replay_parity_rejects_pixels_noise_and_digital_wave_drift():
    rgb=np.zeros((3,2,2));tx=np.ones((4,2));rx=tx*.7
    expected=(rgb.copy(),tx.copy(),rx.copy())
    assert parity(rgb,tx,rx,expected,'raw_adaptive')=={'RGB':0.,'TX':0.,'RX':0.}
    with pytest.raises(ValueError):parity(rgb+1e-3,tx,rx,expected,'wetok_r3')
    with pytest.raises(ValueError):parity(rgb,tx+1e-8,rx,expected,'arithmetic_adaptive')
    with pytest.raises(ValueError):parity(rgb,tx,rx+.001,expected,'wetok_r3')
    with pytest.raises(ValueError):parity(rgb*np.nan,tx,rx,expected,'wetok_r3')


def test_n3060_cannot_bypass_author_or_other_live_worker():
    states=[{'status':s} for s in ['REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING',
        'LOW_BUDGET_MILESTONE_COMPLETE_LATER_STAGES_PENDING','REAL_REFERENCE_METRICS_COMPLETE_PENDING_REVIEW_AND_PUBLICATION',
        'REAL_N4084_REPLAY_COMPLETE_PENDING_REVIEW_AND_PUBLICATION','REAL_AUTHOR_TIMING_COMPLETE_PENDING_REVIEW_AND_PUBLICATION']]
    assert release_ready(states,[True]*4,lambda s:False)
    assert not release_ready(states,[True,True,True,False],lambda s:False)
    assert not release_ready(states,[True]*4,lambda s:s is states[-1])
    states[-1]['status']='WAITING_FOR_N4084_AND_ALL_PREDECESSORS'
    assert not release_ready(states,[True]*4,lambda s:False)
