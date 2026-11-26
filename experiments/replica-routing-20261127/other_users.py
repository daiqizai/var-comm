"""Descriptive nonweak-user quality, derived from the same four output images."""
import sys
from analyze import *

def main(base):
    frozen=json.loads((base/"development/frozen.json").read_text())["policies"]
    specs={k:"/".join(v) for k,v in frozen.items()}
    rows=[]
    for cond in CONDITIONS:
        zz=sum((read(base/"formal"/f"rotated_{cond}_{s}.csv") for s in SEEDS),[])
        for r in zz:
            r["other3"]=(4*float(r["mean_psnr"])-float(r["weak_psnr"]))/3.
        other={name:collect(zz,lambda r:r["spec"]==spec,"other3") for name,spec in specs.items()}
        for a,b in (("best_fixed","latent"),("adaptive","best_fixed"),("full","latent")):
            rows.append(dict(condition=cond,comparison=a+"-"+b,**stat(other[a]-other[b])))
    write(base/"analysis/other_users.csv",rows)

if __name__=="__main__":
    main(Path(sys.argv[1]))
