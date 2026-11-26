"""Secondary comparison, with the same paired quartet bootstrap as the primary."""
from analyze import *

def main(study):
    zz=read(study/"secondary/matched.csv")
    tt=read(study/"secondary/batch1.csv")
    stats=[]
    for cond in CONDITIONS:
        test=[r for r in zz if r["condition"]==cond]
        tm=[r for r in tt if r["condition"]==cond]
        assert len(test)==375 and len(tm)==128
        for met in ("psnr","dists","all_psnr"):
            a=collect(test,lambda r:True,"adaptive_"+met)
            b=collect(test,lambda r:True,"fixed_"+met)
            stats.append(dict(condition=cond,metric=met,comparison="adaptive_minus_fixed",**stat(a-b)))
        diff={(int(v["seed"]),int(v["group"])):float(v["adaptive_choices"]!=v["fixed_choices"]) for v in test}
        va=np.array([[diff[s,g] for g in range(125)] for s in SEEDS]).mean(0)
        stats.append(dict(condition=cond,metric="fraction",comparison="different_active_user_choices",**stat(va)))
        arr={}
        for arm in ("adaptive","fixed"):
            val={(int(v["group"]),int(v["repeat"])):float(v["rx_ms"]) for v in tm if v["arm"]==arm}
            assert len(val)==64
            arr[arm]=np.array([[val[g,r] for r in range(2)] for g in range(32)]).mean(1)
            stats.append(dict(condition=cond,metric="ms",comparison=arm,**stat(arr[arm])))
        stats.append(dict(condition=cond,metric="ratio",comparison="adaptive_div_fixed",**ratio(arr["adaptive"],arr["fixed"])))
    write(study/"analysis/secondary.csv",stats)

if __name__=="__main__":
    import sys
    main(Path(sys.argv[1]))
