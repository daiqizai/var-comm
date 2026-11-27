"""Descriptive complement of the registered weakest-user metric, computed after primary analysis."""
from pathlib import Path
import numpy as np
from analyze import CONDITIONS, DRAWS, read, write, summary


def main():
    root = Path(__file__).resolve().parent
    rows = read(root/"evaluate/group_scores.csv")
    bank = np.random.default_rng(271127).integers(0, 125, (10000, 125))
    out = []
    for cond in CONDITIONS:
        others = {}
        for method in ("pnoma_q2", "semsic_three"):
            result = np.full((len(DRAWS), 125), np.nan)
            for r in rows:
                if r["condition"] == cond and r["method"] == method:
                    g, d = int(r["group"]), DRAWS.index(int(r["draw"]))
                    assert np.isnan(result[d, g])
                    result[d, g] = (4 * float(r["mean_psnr"]) - float(r["weak_psnr"])) / 3
            assert np.isfinite(result).all()
            others[method] = result.mean(axis=0)
        others["semsic_three minus pnoma_q2"] = others["semsic_three"] - others["pnoma_q2"]
        for name, values in others.items():
            out.append(dict(condition=cond, method_or_contrast=name, metric="mean_psnr_other_three",
                            quartets=125, status="exploratory, after primary analysis", **summary(values, bank)))
    write(root/"analysis/other_three.csv", out)

if __name__ == "__main__":
    main()
