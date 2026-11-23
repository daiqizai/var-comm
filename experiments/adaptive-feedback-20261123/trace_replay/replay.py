"""Replay time-stamped handset recordings with the synthetic-fit feedback formats."""
import csv
from collections import Counter
from datetime import datetime
import io
from itertools import groupby
from pathlib import Path, PurePosixPath
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from study import ROOT, TMP, SPEC, digest, record, read, table
from feedback_codes import decode, encode, predict
from field_summary import write_summary

OUT = ROOT / "field"


def segments(rows, counts):
    history, prev = [], None
    for ts, items in groupby(rows, key=lambda r:r.get("Timestamp","").strip()):
        items = list(items)
        counts["raw_rows"] += len(items)
        counts["duplicates"] += len(items)-1
        try:
            now = datetime.strptime(ts,"%Y.%m.%d_%H.%M.%S")
        except ValueError:
            now = None
        val = None
        for r in items:
            try:
                v = float(r["SNR"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and -30. <= v <= 50.:
                val = v
                break
        valid = now is not None and val is not None
        if not valid or (prev is not None and (now-prev).total_seconds() != 1):
            if history:
                yield np.array(history)
            history = []
        if valid:
            history.append(val)
        else:
            counts["invalid_seconds"] += 1
        prev = now if valid else None
    if history:
        yield np.array(history)


def accumulate(obs, slots, target, fz, sums):
    for mode, scheme in SPEC["modes"].items():
        step = fz["models"][mode]["step"]
        codes = encode(obs, scheme, fz["books"], fz["initial_db"], step)
        values = decode(codes, scheme[0], fz["books"], fz["initial_db"], step)
        pred = predict(codes, slots, SPEC, mode, fz)
        last = values[(slots-2)//scheme[1]] if scheme[0] else np.full(len(slots),fz["initial_db"])
        for decoder, y in [("synthetic_fit",pred),("last_decoded",last)]:
            sums[(mode,decoder)] += float(np.square(y-target).sum())
    sums[("unquantized","last_decoded")] += float(np.square(obs[slots-2]-target).sum())


def replay():
    protocol = read(HERE / "protocol.json")
    archive = TMP / protocol["archive"]
    assert digest(archive) == protocol["archive_sha256"]
    fz = read(ROOT / "frozen.json")
    identity = {"external_protocol_sha256":digest(HERE / "protocol.json"), "code_sha256":digest(Path(__file__)),
                "frozen_feedback_sha256":digest(ROOT / "frozen.json"),"archive_sha256":digest(archive)}
    record(OUT / "registration.json", identity)
    rows, coverage = [], []
    with zipfile.ZipFile(archive) as zf:
        names = sorted(s for s in zf.namelist() if s.lower().endswith(".csv") and
                       not any(a.startswith((".", "__")) for a in PurePosixPath(s).parts))
        assert len(names) == 83
        for name in names:
            parts = PurePosixPath(name).parts
            mobility = next(x for x in parts if x in ("Static","Driving"))
            sums, cnt = Counter(), Counter()
            with zf.open(name) as handle:
                stream = io.TextIOWrapper(handle, encoding="utf-8-sig")
                for obs in segments(csv.DictReader(stream), cnt):
                    cnt["segments"] += 1
                    until = 40 + max(0, (len(obs)-40)//4)*4
                    if until == 40:
                        continue
                    cnt["qualifying_segments"] += 1
                    slots = np.arange(40,until)
                    target = obs[slots]
                    cnt["scored_seconds"] += len(slots)
                    cnt["outside_simulated_range"] += int(((target < -5)|(target > 24)).sum())
                    accumulate(obs, slots, target, fz, sums)
            coverage.append(dict(session=name,mobility=mobility, **{k:cnt[k] for k in
                ("raw_rows","duplicates","invalid_seconds","segments","qualifying_segments",
                 "scored_seconds","outside_simulated_range")}))
            for (mode, decoder), value in sorted(sums.items()):
                rows.append(dict(session=name,mobility=mobility,mode=mode,decoder=decoder,
                                 seconds=cnt["scored_seconds"],mse_db2=value/cnt["scored_seconds"]))
    table(OUT / "sessions.csv", rows)
    table(OUT / "coverage.csv", coverage)
    write_summary(OUT, protocol["seed"])


if __name__=="__main__":
    replay()
