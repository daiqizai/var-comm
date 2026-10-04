"""Qualified execution-only health polling; original scientific files untouched.

Healthy hardware checks are sampled at most every five seconds. First checks,
failures and any detected thermal state retain the original check behavior.
This can delay discovery of a newly busy/hot device by up to five seconds plus
one computation boundary; it does not change the original hot-state threshold.
"""
from __future__ import annotations
import time

class HealthyPolling:
    def __init__(self, original, interval=5., clock=time.monotonic):
        self.original=original;self.interval=interval;self.clock=clock
        self.next_check=0.;self.real_checks=0;self.cache_hits=0
        self.timing=dict(calls=0,seconds=0.)
    def __getattr__(self,name):return getattr(self.original,name)
    def check(self):
        now=self.clock()
        if self.original.hot_count==0 and now<self.next_check:
            self.cache_hits+=1;return False
        self.next_check=0.;self.real_checks+=1
        started=self.clock();self.timing['calls']+=1
        try:result=self.original.check()
        finally:self.timing['seconds']+=self.clock()-started
        if not result and self.original.hot_count==0:
            self.next_check=self.clock()+self.interval
        return result

def main():
    import m1_native
    Original=m1_native.Native
    class Native(Original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            probe=self.assets.old.b
            if probe.SAFETY is None:raise RuntimeError('Original runtime safety monitor is absent')
            probe.SAFETY=HealthyPolling(probe.SAFETY)
            self.health_polling=probe.SAFETY
            self.timings['health_query']=self.health_polling.timing
    m1_native.Native=Native
    import m1_run
    m1_run.main()

if __name__=='__main__':main()
