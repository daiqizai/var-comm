"""Synthetic boundary tests only; no real decoder or model is called."""
import unittest
from types import SimpleNamespace
import numpy as np
from public_counter_rx import receive


class FakePHY:
    WIRE_PROTOCOL = 'qualified-wire'
    def __init__(self, profile_id):
        self.profile_id = profile_id
        self.calls = []
    def require(self, condition, message):
        assert condition, message
    def receive_header(self, *args):
        self.calls.append('header')
        return dict(header_ok=self.profile_id is not None, profile_id=self.profile_id)
    def select_received_profile(self, runtime, header):
        return None if not header['header_ok'] else runtime.profiles[str(header['profile_id'])]
    def receive_body(self, runtime, y, profile, snr, counter, *args):
        self.calls.append(('body', profile['profile_id'], counter))
        return dict(status='BODY_REJECT')


def rt():
    keys=('family','mode','m','K','q','nominal_rate','k','n','token_count','profile_key')
    return SimpleNamespace(qualified=True, profiles={str(i):dict(profile_id=i, **dict.fromkeys(keys,0)) for i in (11,29)})


class CounterTests(unittest.TestCase):
    def test_correct_header_matches_old_counter(self):
        p=FakePHY(11);v=receive(p,rt(),np.zeros((1024,2)),10,123000,None,'a')
        self.assertEqual(v['derived_counter'],123011)
        self.assertEqual(p.calls,['header',('body',11,123011)])
    def test_wrong_legal_header_uses_received_profile_without_oracle(self):
        p=FakePHY(29);v=receive(p,rt(),np.zeros((1024,2)),10,123000,None,'a')
        self.assertEqual(v['derived_counter'],123029)
        self.assertEqual(v['rx_profile']['profile_id'],29)
    def test_rejected_header_has_no_body_call(self):
        p=FakePHY(None);v=receive(p,rt(),np.zeros((1024,2)),10,123000,None,'a')
        self.assertEqual(p.calls,['header']);self.assertIsNone(v['derived_counter'])
    def test_invalid_observation_stops_before_header(self):
        p=FakePHY(11)
        with self.assertRaises(AssertionError):receive(p,rt(),np.zeros((1023,2)),10,0,None,'a')
        self.assertEqual(p.calls,[])
    def test_independent_bases(self):
        p=FakePHY(11)
        self.assertEqual(receive(p,rt(),np.zeros((1024,2)),10,4096,None,'a')['derived_counter'],4107)


if __name__=='__main__':unittest.main()
