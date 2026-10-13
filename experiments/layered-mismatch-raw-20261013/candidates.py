"""R0 constructor-only candidate catalogue; never rank, encode, decode, or simulate.

The 433 registered single-group profiles are retained byte-for-byte as JSON values.
New U and EEP profiles share physical IDs when every wire field is equal.  Their
legal dimensions come from the frozen Sionna encoder constructors, not a rate
approximation.  An accepted layout is not a link or reconstruction qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
PREFIX = tuple(sum(s * s for s in SIZES[:m]) for m in range(11))
BODY = 956
N1_VALUES = (256, 384, 512, 640, 768)
Q = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
MOD = {v: k for k, v in Q.items()}
HEADER_KEY = "uep-v1-profile12-crc16-tail6-conv-ratematch136-qpsk-Es2"
REQUEST_SHA = "9e587fcec3be465d8d6c43a7123400635441918befbbd45a6339f0f208af2faf"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def nearest_even(numerator, denominator):
    """Exact integer version of the plan's round(), including ties to even."""
    require(numerator >= 0 and denominator > 0, "Positive rational expected")
    a, b = divmod(numerator, denominator)
    return a + int(2 * b > denominator or (2 * b == denominator and a % 2 == 1))


def frozen_anchors(profiles, policies):
    require(len(profiles) == 433, "The full raw433 authority is required")
    by_candidate = {}
    for p in profiles:
        for cid in set(p.get("alias_candidate_ids", []) + [p["candidate_id"]]):
            if cid in by_candidate:
                require(by_candidate[cid] == p, "Ambiguous frozen candidate alias")
            by_candidate[cid] = p
    require(policies["status"] == "POLICIES_FROZEN_ON_CALIBRATION1000", "Unexpected policy status")
    winners = {int(w["snr_db"]): w["candidate_id"] for w in policies["winners"] if w["family"] == "PARTIAL"}
    rows = []
    for hat in (7, 10):
        for method, table_snr in (("S_nom", hat), ("S_back3", hat - 3)):
            p = by_candidate[winners[table_snr]]
            rows.append(dict(hat_snr_db=hat, method=method, table_snr_db=table_snr,
                             profile_id=p["profile_id"], candidate_id=winners[table_snr],
                             wire_key=p["wire_key"], m=p["m"], K=p["K"],
                             receiver_rule=p["receiver_rule"]))
    return rows


def endpoint_grid(anchors):
    result = {(m, (SIZES[m] ** 2 * part) // 4) for m in (6, 7, 8) for part in range(4)}
    result.update((int(a["m"]), int(a["K"])) for a in anchors)
    for m, k in result:
        require(1 <= m < len(SIZES) and 0 <= k < SIZES[m] ** 2, "Frozen endpoint out of raw range")
    return sorted(result)


def specifications(endpoints):
    for m, k in endpoints:
        for j in (5, 6, 7):
            if j > m or PREFIX[m] + k <= PREFIX[j]:
                continue
            k1, k2 = 12 * PREFIX[j] + 16, 12 * (PREFIX[m] + k - PREFIX[j]) + 16
            for n1 in N1_VALUES:
                for q1 in (2, 4, 6):
                    for q2 in (2, 4, 6):
                        if q1 <= q2:
                            yield dict(family="U", m=m, K=k, j=j, N1=n1, N2=BODY - n1,
                                       q1=q1, q2=q2, k1=k1, k2=k2)
            n1 = nearest_even(BODY * k1, k1 + k2)
            for q in (2, 4, 6):
                yield dict(family="EEP", m=m, K=k, j=j, N1=n1, N2=BODY - n1,
                           q1=q, q2=q, k1=k1, k2=k2)


class LayoutPlanner:
    def __init__(self, backend, unsupported):
        self.backend = backend
        self.unsupported = unsupported
        self.records = {}

    def plan(self, k, n, q):
        key = f"{k}:{n}:{q}"
        if key not in self.records:
            try:
                layout = self.backend.plan(k, n, q)
                require(layout["k"] == k and layout["n"] == n, "Constructor changed information dimensions")
                require(layout["k_ldpc"] - layout["k_filler"] == k, "Filler dimension mismatch")
                require(layout["repetition_bits"] == 0 and n <= layout["n_cb_comp"], "Unsupported repetition")
                self.records[key] = dict(k=k, n=n, q=q, status="LEGAL_CONSTRUCTOR_LAYOUT", layout=layout)
            except self.unsupported as error:
                self.records[key] = dict(k=k, n=n, q=q, status="UNSUPPORTED_BY_FROZEN_BACKEND",
                                         exception_type=type(error).__name__, reason=str(error))
        return self.records[key]


def group(layout, q, symbols):
    source_bits = layout["k"] - 16
    physical = dict(backend=layout["implementation"], source_bits=source_bits, CRC=16,
                    layout=layout, modulation=MOD[q], modulation_padding_bits=0)
    return dict(source_bits=source_bits, crc_bits=16, tail_bits=0, information_bits=layout["k"],
                transmitted_bits=layout["n"], symbols=symbols, modulation=MOD[q],
                modulation_padding_bits=0, layout=layout, phy_key=identity(physical),
                ldpc_k=layout["k_ldpc"], filler_bits=layout["k_filler"], mother_bits=layout["mother_bits"],
                effective_information_rate=layout["k"] / layout["n"],
                source_rate=source_bits / layout["n"], nominal_rate=None,
                mcs_id=f"{MOD[q]}:actual_{layout['k']}/{layout['n']}")


def generate(planner, endpoints):
    physical, screening = {}, []
    for spec in specifications(endpoints):
        row = dict(spec)
        bad = [g for g in (1, 2) if 6 * spec[f"k{g}"] > 5 * spec[f"q{g}"] * spec[f"N{g}"]]
        if bad:
            row.update(status="REJECT_MAX_RATE_5_OVER_6", rejected_groups=bad)
            screening.append(row)
            continue
        attempts = [planner.plan(spec[f"k{g}"], spec[f"q{g}"] * spec[f"N{g}"], spec[f"q{g}"]) for g in (1, 2)]
        if any(a["status"] != "LEGAL_CONSTRUCTOR_LAYOUT" for a in attempts):
            row.update(status="REJECT_BACKEND", group_admission=[{k: v for k, v in a.items() if k != "layout"} for a in attempts])
            screening.append(row)
            continue
        groups = [group(a["layout"], spec[f"q{i}"], spec[f"N{i}"]) for i, a in enumerate(attempts, 1)]
        wire = dict(N=1024, G=2, j=spec["j"], m=spec["m"], K=spec["K"], order="raster",
                    header_phy_key=HEADER_KEY, header_symbols=68, idle_symbols=0,
                    receiver_rule="DROP", group_phy_keys=[g["phy_key"] for g in groups])
        key = identity(wire)
        if key not in physical:
            physical[key] = dict(**{k: v for k, v in wire.items() if k != "group_phy_keys"},
                                 groups=groups, wire_key=key, candidate_id=key, stable_id=key,
                                 token_count=PREFIX[spec["m"]] + spec["K"], base_token_count=PREFIX[spec["j"]],
                                 body_symbols=956, used_body_symbols=956,
                                 header_information_bits=12, header_crc_bits=16, header_tail_bits=6,
                                 header_transmitted_bits=136, full_state_id=f"m{spec['m']}_K{spec['K']}",
                                 base_state_id=f"m{spec['j']}_K0", gray_state_id="gray",
                                 family_memberships=[], allocation_modes=[], profile_id=None,
                                 new_link_roundtrip_qualified=False, ranking_status="NOT_RANKED")
        p = physical[key]
        p["family_memberships"].append(spec["family"])
        p["allocation_modes"].append("finite_N1_grid" if spec["family"] == "U" else "round_nearest_even_information_proportional")
        p["family_memberships"] = sorted(set(p["family_memberships"]))
        p["allocation_modes"] = sorted(set(p["allocation_modes"]))
        row.update(status="LEGAL_CONSTRUCTOR_LAYOUT", wire_key=key)
        screening.append(row)
    return [physical[k] for k in sorted(physical)], screening


def public_catalogue(old, new):
    require([p["profile_id"] for p in old] == list(range(len(old))), "Existing public IDs changed")
    keys = [p["wire_key"] for p in old + new]
    require(len(keys) == len(set(keys)), "Physical identities overlap")
    profiles = copy.deepcopy(old + new)
    full_fit = len(profiles) <= 4096
    if full_fit:
        for index, p in enumerate(profiles[len(old):], len(old)):
            p["profile_id"] = index
    return dict(status="CPU_PROPOSED_PUBLIC_DOMAIN_REQUIRES_SCIENTIFIC_REGISTRATION" if full_fit else "FULL_SUPERSET_EXCEEDS_12BIT_REGISTER_ACTUAL_SUBSET_BEFORE_USE",
                profiles=profiles, profile_count=len(profiles), original_single_profile_count=len(old),
                full_superset_fits_12bit=full_fit, public_profile_id_bits=12,
                catalogue_digest=identity(profiles),
                overflow_policy="Never truncate candidates or reuse IDs; freeze actual nominated subset if full domain exceeds 4096.",
                single_group_logical_receivers=["KEEP", "DROP"],
                original_single_receiver_rule_unchanged=True,
                single_rule_note="S_nom and S_back3 retain original KEEP. S_robust KEEP/DROP variants share each physical observation; rule is separately frozen after calibration.",
                two_group_receiver_rule="longest_contiguous_crc_accepted_prefix_or_gray",
                grouped_receiver_domain_rule="U/EEP receiver always DROP, including a wrong but legal received raw433 G1 ID. Old row receiver_rule is format provenance, not a KEEP instruction to the grouped receiver.",
                original_single_receiver_domain="Original raw433 only; do not extend S_nom/S_back3/S_robust to new G2 IDs.",
                nomination_status="NOT_RANKED_NO_BLER_OR_RECONSTRUCTION_CALLS")


def load_module(path, name):
    require(name not in sys.modules, f"Unexpected pre-imported module {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def forbidden(*args, **kwargs):
    raise RuntimeError("R0 forbids encode/decode/packet/channel/model calls")


def load_frozen_backend(request):
    require(sha(request) == REQUEST_SHA, "Unexpected authority request bytes")
    data = json.loads(Path(request).read_text())
    binding = data["raw_runtime_binding"]
    pins = {f["path"]: f["sha256"] for f in binding["files"]}
    used = []
    roles = binding["roles"]
    def checked(path):
        path = str(path)
        require(path in pins and sha(path) == pins[path], f"Frozen source mismatch: {path}")
        used.append(dict(path=path, sha256=pins[path]))
        return Path(path)
    modules = {}
    for role, name in (("original_common", "uep_common"), ("original_planner", "profiles"),
                       ("original_backend", "ldpc_backend"), ("plan_module", "main_raw64_plan_only"),
                       ("adapter_module", "main_raw64_packet_adapter")):
        modules[name] = load_module(checked(roles[role]), name)
    backend = modules["ldpc_backend"].SionnaBackend(device="cpu", qualification=str(checked(roles["legacy_qualification"])))
    require(backend.qualified, "Original LDPC implementation identity not qualified")
    torch = backend.torch
    require(not torch.cuda.is_initialized(), "CUDA already initialized in R0")
    modules["main_raw64_packet_adapter"].runtime_flags(torch)
    backend.encode = backend.decode = backend.codecs = forbidden
    backend.Encoder.forward = forbidden
    backend.Decoder.__init__ = forbidden
    adapter = modules["main_raw64_packet_adapter"].Backend(backend)
    adapter.encode = adapter.decode = forbidden
    cat = json.loads(checked(roles["catalogue"]).read_text())
    modules["main_raw64_packet_adapter"].validate_catalogue(cat, backend.identity)
    policy_path = next(f["path"] for f in binding["files"] if f["original_path"].endswith("/final_selection_v1/policies.json"))
    policies = json.loads(checked(policy_path).read_text())
    planner = LayoutPlanner(adapter, (modules["profiles"].UnsupportedConfiguration,
                                      modules["main_raw64_plan_only"].UnsupportedConfiguration))
    return planner, cat, policies, used, torch


def csv_rows(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def run(args):
    out = Path(args.out).resolve()
    require(not out.exists(), "Immutable R0 output already exists; inspect completion or failure instead of overwriting")
    out.mkdir(parents=True)
    start = time.time()
    save(out / "launch.json", dict(pid=os.getpid(), start_unix=start, command=sys.argv,
         authority_request=dict(path=str(Path(args.request).resolve()), sha256=sha(args.request)),
         source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
         authorized_operation="CPU encoder-constructor metadata only", scientific_calls=0))
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "R0 must hide CUDA devices explicitly")
        if hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        planner, old_cat, policies, inputs, torch = load_frozen_backend(args.request)
        old = old_cat["profiles"]
        require(len(old) == 433, "All raw433 profiles are mandatory")
        # Reconstruct every original layout without changing any original field.
        for p in old:
            for g in p["groups"]:
                rec = planner.plan(g["information_bits"], g["transmitted_bits"], Q[g["modulation"]])
                require(rec["status"] == "LEGAL_CONSTRUCTOR_LAYOUT" and rec["layout"] == g["layout"],
                        f"Existing profile {p['profile_id']} no longer has its exact frozen layout")
        anchors = frozen_anchors(old, policies)
        endpoints = endpoint_grid(anchors)
        new, screening = generate(planner, endpoints)
        catalogue = public_catalogue(old, new)
        require(catalogue["profiles"][:433] == old, "Original raw433 values were changed")
        require(not torch.cuda.is_initialized(), "R0 unexpectedly initialized CUDA")
        save(out / "catalogue.json", catalogue)
        save(out / "group_layouts.json", planner.records)
        save(out / "anchors.json", anchors)
        save(out / "input_bindings.json", inputs)
        groups = {g["phy_key"]: g for p in catalogue["profiles"] for g in p["groups"]}
        save(out / "groups_by_phy_key.json", groups)
        csv_rows(out / "screening.csv", screening)
        flat = []
        for p in catalogue["profiles"]:
            r = {k: p[k] for k in ("profile_id", "wire_key", "m", "K", "j", "G", "token_count", "receiver_rule", "family_memberships")}
            for i, g in enumerate(p["groups"], 1):
                r.update({f"g{i}_{k}": v for k, v in dict(k=g["information_bits"], n=g["transmitted_bits"], q=Q[g["modulation"]],
                     symbols=g["symbols"], phy_key=g["phy_key"], bg=g["layout"]["bg"], lifting=g["layout"]["z"],
                     k_ldpc=g["layout"]["k_ldpc"], filler=g["layout"]["k_filler"], rate=g["effective_information_rate"],
                     n_cb_comp=g["layout"]["n_cb_comp"], first_2Z_punctured=g["layout"]["puncturing_first_2Z"],
                     remaining_punctured=g["layout"]["puncturing_remaining"], repetition_bits=g["layout"]["repetition_bits"]).items()})
            flat.append(r)
        csv_rows(out / "candidates.csv", flat)
        counts = dict(Counter(r["family"] + ":" + r["status"] for r in screening))
        report = dict(status="PASS_R0_CONSTRUCTOR_ONLY_NOT_LINK_QUALIFICATION", original_raw433_retained=433,
                      proposed_new_physical_profiles=len(new), proposed_total_profiles=len(catalogue["profiles"]),
                      screening_counts=counts, endpoint_grid=[dict(m=m, K=k) for m, k in endpoints],
                      U_profile_count=sum("U" in p["family_memberships"] for p in new),
                      EEP_profile_count=sum("EEP" in p["family_memberships"] for p in new),
                      layout_query_count=len(planner.records), layout_status_counts=dict(Counter(r["status"] for r in planner.records.values())),
                      public_domain_fits_12bit=catalogue["full_superset_fits_12bit"],
                      catalogue_digest=catalogue["catalogue_digest"], cuda_initialized=False,
                      encoder_forward_calls=0, decoder_calls=0, channel_calls=0, packet_calls=0, model_calls=0,
                      ranking_status="NOT_RANKED_BLER_COVERAGE_MUST_BE_CHECKED_BY_EXACT_PHY_IDENTITY",
                      old_grouped_q6_gap="Old FrameLink maps all non-QPSK to 4 bits. New grouped endpoint must dispatch q=6 explicitly via frozen raw64 modulation/demap; no packet qualification performed here.",
                      EEP_note="Equal modulation and information-proportional symbols rounded ties-to-even; actual group rates reported. Equal BLER is not assumed.",
                      source_sha256=sha(__file__), elapsed_seconds=time.time() - start)
        save(out / "report.json", report)
        save(out / "completion.json", dict(status=report["status"], output_files=[dict(path=p.name, bytes=p.stat().st_size,
             sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()], science_calls=0))
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        save(out / "failure.json", dict(status="FAILED_NO_AUTOMATIC_RETRY", error_type=type(error).__name__,
             error=str(error), traceback=traceback.format_exc(), science_calls=0))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    run(parser.parse_args())
