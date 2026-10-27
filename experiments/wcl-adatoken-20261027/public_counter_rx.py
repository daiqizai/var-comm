"""Derive the scrambler counter only after decoding the actual paid header.

public_base identifies the frame independently of the transmitter's profile.
All numerical PHY functions are the unchanged qualified implementations.
"""
import numpy as np


def receive(phy, runtime, observed, snr_db, public_base, ledger, event_id,
            phase='development2'):
    assert type(public_base) is int and public_base >= 0
    assert isinstance(event_id, str) and event_id
    assert runtime.qualified
    y = np.asarray(observed)
    phy.require(y.shape == (1024, 2) and np.isfinite(y).all(),
                'Exactly1024 finite received symbols')
    header = phy.receive_header(runtime, y[:68], snr_db, ledger,
                                event_id + ':header', phase)
    p = phy.select_received_profile(runtime, header)
    if p is None:
        return dict(header=header, body=None, rx_profile=None, status='HEADER_REJECT',
                    public_base=public_base, derived_counter=None)
    profile_id = p['profile_id']
    assert type(profile_id) is int and 0 <= profile_id < 4096
    counter = public_base + profile_id
    body = phy.receive_body(runtime, y[68:], p, snr_db, counter, ledger,
                            event_id + ':body', phase, phy.WIRE_PROTOCOL)
    keys = ('profile_id', 'family', 'mode', 'm', 'K', 'q', 'nominal_rate',
            'k', 'n', 'token_count', 'profile_key')
    return dict(header=header, body=body, rx_profile={k: p[k] for k in keys},
                status=body['status'], public_base=public_base, derived_counter=counter)
