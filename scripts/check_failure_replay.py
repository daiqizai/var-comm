#!/usr/bin/env python3
"""Check first-failure retention without models, source truth, or a channel."""

from copy import deepcopy
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from var_comm.failure_replay import SIZES, choose_oracle, image_only_state
from var_comm.study import artifact_hashes, create_output, snapshot, write_json


def fixture(failure):
    scales = [[index + 1] * size ** 2 for index, size in enumerate(SIZES[:9])]
    blocks = [sum(scales[:6], []), *scales[6:9]]
    events = [{"group": index, "accepted": index != failure, "recovered_tokens": block}
              for index, block in enumerate(blocks) if failure is None or index <= failure]
    trusted = 9 if failure is None else (0 if failure == 0 else failure + 5)
    return {"header": {"accepted": True, "mode": 9, "label": 12}, "label": 12,
            "output_prefix": scales[:trusted], "last_accepted_scale": trusted, "events": events}


def main():
    cases = 0
    for failure in (0, 1, 2, 3, None):
        record = fixture(failure)
        before = deepcopy(record)
        state = image_only_state(record)
        assert record == before
        assert state["trusted_prefix"] == before["output_prefix"]
        assert state["render_scales"] == (9 if failure is None else 6 + failure)
        assert state["retention_applied"] == (failure is not None)
        state["render_prefix"][0][0] = 4000
        assert record == before and state["trusted_prefix"] == before["output_prefix"]
        cases += 1
    for accepted, mode in ((False, 9), (True, 8)):
        record = {"header": {"accepted": accepted, "mode": mode, "label": 12}, "label": None,
                  "output_prefix": [], "last_accepted_scale": 0, "events": []}
        assert image_only_state(record)["failure_stage"] == "header_failure"
        cases += 1
    for values in ([], [1], [4096] * 64, [-1] * 64, [1.0] * 64):
        record = fixture(1)
        record["events"][-1]["recovered_tokens"] = values
        state = image_only_state(record)
        assert not state["candidate_available"] and state["render_prefix"] == record["output_prefix"]
        cases += 1
    record = fixture(1)
    record["events"].append({"group": 2, "accepted": True, "recovered_tokens": [1] * 100})
    try:
        image_only_state(record)
    except ValueError:
        cases += 1
    else:
        raise AssertionError("continued decoding after failure was accepted")
    record = fixture(2)
    record["output_prefix"][0][0] = 1234
    try:
        image_only_state(record)
    except ValueError:
        cases += 1
    else:
        raise AssertionError("changed trusted state was accepted")
    before = {"family": "group_var", "lpips_alex": 0.2, "psnr_db": 30.0, "policy": "discard"}
    after = {"family": "group_var", "lpips_alex": 0.1, "psnr_db": 20.0, "policy": "retain"}
    chosen = choose_oracle(before, after)
    assert chosen["psnr_db"] == 20.0 and not chosen["deployable"]
    assert choose_oracle(before, dict(after, lpips_alex=0.2))["oracle_selected_policy"] == "discard"
    cases += 2
    output = create_output(ROOT / "outputs/VAR-CRC-FAILURE-REPLAY-SELFCHECK-001")
    sources = snapshot(output, [Path(__file__), ROOT / "src/var_comm/failure_replay.py", ROOT / "src/var_comm/study.py"])
    result = {"status": "FAILURE_REPLAY_SELFCHECK_PASS", "cases": cases, "source_hashes": sources,
              "no_models_or_ground_truth": True, "output_hashes": artifact_hashes(output)}
    write_json(output / "selfcheck.json", result)
    print(result["status"], cases, flush=True)


if __name__ == "__main__":
    main()
