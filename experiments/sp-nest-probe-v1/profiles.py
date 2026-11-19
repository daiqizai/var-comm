"""Public superposition directory. No training or image-dependent pruning."""
from fractions import Fraction
from itertools import product
from collections import defaultdict

SIDES=(1,2,3,4,5,6,8,10,13,16)
OFF=tuple(sum(v*v for v in SIDES[:i]) for i in range(11))
QUARTER=tuple((m,k) for m in range(4,10) for k in (0,*(i*SIDES[m]**2//4 for i in (1,2,3))))+((10,0),)
BR=('1/3','1/2','2/3'); EPHY=((2,'2/3'),(4,'1/2'),(4,'2/3'),(4,'3/4'),(6,'2/3'))
ALPHA=(75,80,85,90); N=1024; NBODY=956

def count(e): return OFF[e[0]]+e[1]
def capacity(q,r): return NBODY*q*Fraction(r).numerator//Fraction(r).denominator-29

def bit_bound(n):
    # TOTAL=2^24 positive counts; 32-bit range; after renorm range width
    # exceeds 2^30, so an interval for one symbol has width >=64. At most
    # 26 doublings per symbol; emitted + pending + 2 is the terminated length.
    return 26*n+2

def reachable_superset():
    rows=set(); naive=set(); witness=defaultdict(set)
    for jcap,br,(qe,er),alpha in product((5,6,7),BR,EPHY,ALPHA):
        cb,ce=capacity(2,br),capacity(qe,er)
        for gridname,grid in [('QUARTER',QUARTER),('WHOLE',tuple(e for e in QUARTER if e[1]==0))]:
            for target in [e for e in grid if e[0]>=6 and count(e)>OFF[jcap]]:
                for j in range(4,jcap+1):
                    eligible=[e for e in grid if OFF[j]<count(e)<=count(target)]
                    ends=[(j,0)]+eligible
                    for m,k in ends: naive.add((j,m,k,br,qe,er,alpha))
                    if any(bit_bound(OFF[t])<=cb for t in range(j+1,jcap+1)): continue
                    floor=max([OFF[j]]+[count(e) for e in eligible if bit_bound(count(e)-OFF[j])<=ce])
                    for m,k in ends:
                        if count((m,k))>=floor:
                            row=(j,m,k,br,qe,er,alpha); rows.add(row); witness[row].add(gridname)
    assert len(rows)+1<=4096, len(rows)+1
    return tuple(sorted(rows)),len(naive)+1,witness

def public_directory():
    rows,_,_=reachable_superset()
    return {0:None,**{i+1:row for i,row in enumerate(rows)}}

if __name__=='__main__':
    rows,n,_=reachable_superset();print('unpruned including NULL:',n,'public including NULL:',len(rows)+1)
