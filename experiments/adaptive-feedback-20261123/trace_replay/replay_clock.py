"""Calendar-time replay. Short holes hold a past measurement for encoding, never for scoring."""
import csv
from collections import Counter
from datetime import datetime
from itertools import groupby
import io
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

from replay import HERE, ROOT, TMP, accumulate, digest, read, record, table
from field_summary import write_summary


def clock_segments(rows, counts):
    vals, measured, prev = [], [], None
    for ts, items in groupby(rows, key=lambda r:r.get("Timestamp","").strip()):
        items = list(items)
        counts["raw_rows"] += len(items)
        counts["duplicates"] += len(items)-1
        try:
            now = datetime.strptime(ts,"%Y.%m.%d_%H.%M.%S")
        except ValueError:
            counts["invalid_seconds"] += 1
            if vals:
                yield np.asarray(vals), np.asarray(measured)
            vals, measured, prev = [], [], None
            continue
        val = None
        for row in items:
            try:
                item = float(row["SNR"])
            except (TypeError,ValueError):
                continue
            if np.isfinite(item) and -30. <= item <= 50.:
                val = item
                break
        if val is None:
            counts["invalid_seconds"] += 1
            continue
        gap = int((now-prev).total_seconds()) if prev is not None else 1
        if gap < 1 or gap > 3:
            yield np.asarray(vals), np.asarray(measured)
            vals, measured = [], []
        else:
            for _ in range(gap-1):
                vals.append(vals[-1])
                measured.append(False)
                counts["held_encoder_seconds"] += 1
        vals.append(val)
        measured.append(True)
        prev = now
    if vals:
        yield np.asarray(vals), np.asarray(measured)


def main():
    out = ROOT / "field_clock"
    protocol = read(HERE / "protocol.json")
    archive = TMP / protocol["archive"]
    assert digest(archive) == protocol["archive_sha256"]
    fz = read(ROOT / "frozen.json")
    record(out / "registration.json", {"archive_sha256": digest(archive),
           "frozen_feedback_sha256": digest(ROOT/"frozen.json"), "strict_coverage_sha256":digest(ROOT/"field/coverage.csv"),
           "protocols_and_code": {n:digest(HERE/n) for n in ["clock_protocol.json","protocol.json",
                                                              "replay_clock.py","replay.py","field_summary.py"]}})
    rows, coverage = [], []
    with zipfile.ZipFile(archive) as zf:
        names = sorted(s for s in zf.namelist() if s.lower().endswith(".csv") and
                       not any(a.startswith((".","__")) for a in PurePosixPath(s).parts))
        assert len(names) == 83
        for name in names:
            mobility = next(x for x in PurePosixPath(name).parts if x in ("Static","Driving"))
            cnt, totals = Counter(), Counter()
            with zf.open(name) as handle:
                stream = io.TextIOWrapper(handle, encoding="utf-8-sig")
                for obs, valid in clock_segments(csv.DictReader(stream), cnt):
                    cnt["segments"] += 1
                    groups = max(0, (len(obs)-40)//4)
                    slots = np.arange(40,40+groups*4).reshape(-1,4)
                    slots = slots[valid[slots].all(axis=1)].ravel()
                    if not len(slots):
                        continue
                    assert valid[slots].all()
                    cnt["qualifying_segments"] += 1
                    cnt["scored_seconds"] += len(slots)
                    target = obs[slots]
                    cnt["outside_simulated_range"] += int(((target < -5)|(target > 24)).sum())
                    accumulate(obs, slots, target, fz, totals)
            coverage.append(dict(session=name,mobility=mobility, **{k:cnt[k] for k in
                ("raw_rows","duplicates","invalid_seconds","segments","qualifying_segments",
                 "scored_seconds","outside_simulated_range","held_encoder_seconds")}))
            for (mode, decoder), value in sorted(totals.items()):
                rows.append(dict(session=name,mobility=mobility,mode=mode,decoder=decoder,
                                 seconds=cnt["scored_seconds"],mse_db2=value/cnt["scored_seconds"]))
    table(out / "sessions.csv", rows)
    table(out / "coverage.csv", coverage)
    write_summary(out, protocol["seed"])


if __name__ == "__main__":
    main()
