"""Read-only exact join to the archived BLER table; no probability imputation."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import shutil

W = Path(__file__).resolve().parents[2]
O = W / "results/layered_mismatch_raw_20261013/candidates_cpu_v1"
D = O / "coverage_v2"
D.mkdir()
S = W / ".research/parallel_v2_20261005/reference/snapshot"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def save(name, v):
    with (D / name).open("x", encoding="utf-8") as f: json.dump(v, f, sort_keys=True, indent=2)


manifest = json.loads((S / "snapshot_manifest.json").read_text())
for name in ("bler.json", "quality_bundle.json", "quality_registration.json", "quality_completion.json"):
    assert sha(S / name) == manifest["entries"][name]["sha256"], name
b = json.loads((S / "bler.json").read_text())
q = json.loads((S / "quality_bundle.json").read_text())
cat = json.loads((O / "catalogue.json").read_text())
lookup = {(r["phy_key"], int(r["snr_db"])): (i, r) for i, r in enumerate(b["rows"])}
assert len(lookup) == len(b["rows"])
qsources = q["source_ids"]
assert len(qsources) == 1000 and len(set(qsources)) == 1000
assert qsources == q["calibration_source_ids"]
qstates = {r["state_id"] for r in q["rows"] if r["receiver"] in ("VAR", "gray")}
rows, matches = [], {}
for p in cat["profiles"]:
    fam = "S" if p["G"] == 1 else "/".join(p["family_memberships"])
    for hat in (7, 10):
        missing = []
        exact = []
        for gi, g in enumerate(p["groups"], 1):
            for snr in (hat - 3, hat, hat + 3):
                found = lookup.get((g["phy_key"], snr))
                if found is None:
                    missing.append(dict(group=gi, snr_db=snr, phy_key=g["phy_key"],
                                        q={"QPSK":2,"16QAM":4,"64QAM":6}[g["modulation"]],
                                        k=g["information_bits"], n=g["transmitted_bits"],
                                        reason="EXACT_PHY_KEY_AND_SNR_ABSENT"))
                    continue
                index, r = found
                pc = r["physical_configuration"]
                for key in ("phy_key", "source_bits", "crc_bits", "tail_bits", "information_bits",
                            "transmitted_bits", "modulation", "layout"):
                    assert pc[key] == g[key], (p["profile_id"], key)
                assert r["code_impl"] == g["layout"]["implementation"]
                rec = {key:r[key] for key in ("phy_key", "snr_db", "n_blocks", "n_correct", "n_reject",
                       "n_undetected", "p_correct", "p_reject", "p_undetected", "p_correct_ci",
                       "p_reject_ci", "p_undetected_ci", "applicability")}
                rec.update(original_row_index=index, physical_configuration_sha256=identity(pc))
                matches[(g["phy_key"], snr)] = rec
                exact.append(dict(group=gi, snr_db=snr, original_row_index=index))
        states = ["gray", f"m{p['m']}_K{p['K']}"]
        if p["G"] == 2: states.append(f"m{p['j']}_K0")
        state_status = {state:("ABSENT_FROM_OLD_Q_TABLE" if state not in qstates else
                            "WRONG_POSITIVE_K_ORDER_ENTROPY_NOT_RASTER" if state not in ("gray",) and not state.endswith("_K0") else
                            "K0_OR_GRAY_PRESENT_MODEL_SOURCE_VALIDATION_PENDING") for state in states}
        rows.append(dict(profile_id=p["profile_id"], wire_key=p["wire_key"], family=fam, G=p["G"],
              m=p["m"], K=p["K"], j=p["j"], hat_snr_db=hat,
              full_body_bler_coverage=not missing, matched_body_snr_rows=len(exact), missing=missing,
              group_modulations=[g["modulation"] for g in p["groups"]],
              clean_quality_state_status=state_status,
              full_quality_compatible_state_names=all(v.startswith("K0_OR_GRAY_PRESENT") for v in state_status.values()),
              J_mean=None, J_interval=None, rank=None,
              nomination_status="NOT_NOMINATED_MISSING_RAW_QUALITY_OR_FULL_FAIR_DOMAIN_EVIDENCE"))
headers = []
for snr in (4,7,10,13):
    index, r = lookup[("uep-v1-profile12-crc16-tail6-conv-ratematch136-qpsk-Es2",snr)]
    headers.append(dict(original_row_index=index, row=r))
counts = {}
for family in ("S", "U", "EEP"):
    scope = [r for r in rows if r["family"] == family and r["hat_snr_db"] == 7]
    counts[family] = dict(total=len(scope), exact_all_groups_all_snr=sum(r["full_body_bler_coverage"] for r in scope),
                         missing=len([r for r in scope if not r["full_body_bler_coverage"]]),
                         complete_bler_and_K0_gray_state_names=sum(r["full_body_bler_coverage"] and r["full_quality_compatible_state_names"] for r in scope))
report = dict(status="EXACT_COVERAGE_AUDIT_COMPLETE_NO_R1_NOMINATION", counts=counts,
   metadata_revision="Gray output is registered under receiver='gray', not receiver='VAR'. This revision corrects the quality-state-name coverage only; all exact BLER coverage counts are unchanged.",
   source_bindings={str(S/name):sha(S/name) for name in ("snapshot_manifest.json", "bler.json", "quality_bundle.json", "quality_registration.json", "quality_completion.json")},
   candidates_catalogue_sha256=sha(O/"catalogue.json"), input_source_sha256=sha(__file__),
   bler_rows=len(b["rows"]), actual_bler_snr=sorted({r["snr_db"] for r in b["rows"]}),
   matched_exact_body_snr_rows=len(matches), source_order_count=len(qsources), source_order_sha256=identity(qsources),
   table_source_probability_mode=b["source_probability_mode"],
   table_real_image_payload_validation_complete=b["real_image_payload_probability_validation_complete"],
   table_conditional_group_independence=b["conditional_group_independence"],
   usable_quality_note="K0/gray names alone do not establish model/preprocessing/source identity. Positive-K rows are old entropy ordering, excluded for raw raster. Separate source/model audit is required.",
   unknown_policy="Missing BLER is null, never 0/1 or nearest-SNR interpolation. No fair total ranking over 433 single and all grouped profiles is identified by these data.",
   missing_q6_single_profiles=sum(p["G"]==1 and p["groups"][0]["modulation"]=="64QAM" for p in cat["profiles"]),
   all_candidate_privileges_retained=True, new_physical_frames=0, bootstrap_calls=0,
   matched_header_rows=headers)
save("coverage_rows.json", rows)
save("matched_bler_rows.json", list(matches.values()))
save("coverage_report.json", report)
with (D / "coverage.csv").open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
    for r in rows: writer.writerow({k:json.dumps(v,sort_keys=True,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in r.items()})
shutil.copyfile(__file__, D / "coverage_audit.py")
save("coverage_completion.json", dict(status=report["status"], files=[dict(path=name, sha256=sha(D/name), bytes=(D/name).stat().st_size) for name in
    ("coverage_rows.json","matched_bler_rows.json","coverage_report.json","coverage.csv","coverage_audit.py")], actual_exit_code=0, science_calls=0))
print(json.dumps({k:v for k,v in report.items() if k not in ('matched_header_rows','source_bindings')},sort_keys=True))
