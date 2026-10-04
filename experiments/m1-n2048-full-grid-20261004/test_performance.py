import unittest
from m1_performance import HealthyPolling

class Monitor:
    def __init__(self):self.hot_count=0;self.calls=0;self.error=False
    def check(self):
        self.calls+=1
        if self.error:raise RuntimeError('query failed')
        return self.hot_count>=3

class PerformanceChecks(unittest.TestCase):
    def test_first_check_and_healthy_interval(self):
        m=Monitor();clock=[10.];p=HealthyPolling(m,clock=lambda:clock[0])
        self.assertFalse(p.check());self.assertFalse(p.check());self.assertEqual(m.calls,1)
        clock[0]=15.;self.assertFalse(p.check());self.assertEqual(m.calls,2)
    def test_detected_hot_is_never_cached(self):
        m=Monitor();p=HealthyPolling(m,clock=lambda:10.)
        p.check();m.hot_count=1;p.check();m.hot_count=2;p.check();m.hot_count=3
        self.assertTrue(p.check());self.assertEqual(m.calls,4)
    def test_failed_query_propagates_and_is_retried(self):
        m=Monitor();m.error=True;p=HealthyPolling(m,clock=lambda:10.)
        for _ in range(2):
            with self.assertRaises(RuntimeError):p.check()
        self.assertEqual(m.calls,2);self.assertEqual(p.cache_hits,0)
    def test_unhealthy_return_is_not_cached(self):
        m=Monitor();m.hot_count=3;p=HealthyPolling(m,clock=lambda:10.)
        self.assertTrue(p.check());self.assertTrue(p.check());self.assertEqual(m.calls,2)

if __name__=='__main__':unittest.main()
