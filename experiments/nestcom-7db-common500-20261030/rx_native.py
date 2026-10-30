"""Invoke the closed N512 received-only mathematical route with a new request."""
from g import *
import rx_worker_v2 as original
original.L=L
if __name__=='__main__':original.main(*sys.argv[1:])
