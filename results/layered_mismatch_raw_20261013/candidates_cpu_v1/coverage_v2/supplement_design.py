"""Quantify missing R0 data without selecting a candidate or executing science."""
from pathlib import Path
import hashlib
import json

W = Path(__file__).resolve().parents[2]
O = W / "results/layered_mismatch_raw_20261013/candidates_cpu_v1"
D = O / "coverage_v2"
S = W / ".research/parallel_v2_20261005/reference/snapshot"
cat = json.loads((O / "catalogue.json").read_text())["profiles"]
anchors = json.loads((O / "anchors.json").read_text())
blr = json.loads((S / "bler.json").read_text())
q = json.loads((S / "quality_bundle.json").read_text())
known = {(r["phy_key"], int(r["snr_db"])) for r in blr["rows"]}
qstates = {r["state_id"] for r in q["rows"] if r["receiver"] in ("VAR", "gray")}
groups = {}
reports = {}
for family in ("ALL", "S", "U", "EEP", "anchors"):
    selected = [p for p in cat if family == "ALL" or (family == "S" and p["G"] == 1)
                or family in p["family_memberships"] or (family == "anchors" and p["profile_id"] in {a["profile_id"] for a in anchors})]
    end = sorted({(p["m"], p["K"]) for p in selected})
    states = sorted(set(end) | {(p["j"], 0) for p in selected if p["G"] == 2})
    gs = {g["phy_key"]:g for p in selected for g in p["groups"]}
    missing = [dict(phy_key=key, snr_db=snr, q={"QPSK":2,"16QAM":4,"64QAM":6}[g["modulation"]],
                    k=g["information_bits"], n=g["transmitted_bits"], layout_id=g["layout"]["layout_id"])
               for key,g in gs.items() for snr in (4,7,10,13) if (key,snr) not in known]
    missing_states = [(m,k) for m,k in states if k > 0 or f"m{m}_K0" not in qstates]
    reports[family] = dict(profile_count=len(selected), endpoints=[dict(m=m,K=k) for m,k in end],
        endpoint_count=len(end), K0_count=sum(k==0 for m,k in end), positiveK_count=sum(k>0 for m,k in end),
        reconstruction_state_count_including_base=len(states),
        K0_names_present=[dict(m=m,K=k) for m,k in states if k==0 and f"m{m}_K0" in qstates],
        absent_K0_names=[dict(m=m,K=k) for m,k in states if k==0 and f"m{m}_K0" not in qstates],
        currently_missing_raw_clean_states=[dict(m=m,K=k) for m,k in missing_states],
        conditional_cal200_new_clean_count=200*len(missing_states), unique_group_layouts=len(gs),
        four_snr_layout_points=4*len(gs), known_exact_layout_snr_points=4*len(gs)-len(missing),
        missing_exact_layout_snr_points=len(missing), missing_points=missing)
value = dict(status="DESIGN_ONLY_NO_SUPPLEMENT_AUTHORIZED_OR_STARTED", families=reports,
    qualification_dependency="K0/gray reuse counts are conditional on independent renderer/metric/source identity check; final data may lower missing counts if separately audited raw-clean caches exist.",
    fair_nomination_identified=False,
    blocking_reason="The only two U profiles with complete old BLER (531 and 573) both have positive K. Their raster clean quality is absent; 199 other U profiles and all 163 q6 single profiles cannot be excluded fairly from unknown BLER/quality.",
    narrow_known_subset=dict(U_profile_ids=[531,573], clean_endpoints=[dict(m=7,K=25),dict(m=6,K=48)],
          cal200_clean_calls=400, new_bler_points=0,
          anchors_additional_clean_endpoints=[dict(m=6,K=16),dict(m=7,K=75)],
          including_anchors_conditional_cal200_clean_calls=800,
          tradeoff="A table-coverage-restricted diagnostic only. It drops 199 U alternatives, 247 single alternatives from BLER screening including all q6, and cannot claim full-domain robust selection or satisfy equal opportunity without an explicit scope change."),
    possible_next_choices=[
        dict(name="Explicit predeclared limited screening scope", action="Define symmetric geometry/coverage rules before viewing new quality and explicitly approve separate capped BLER and clean-Q budgets.",
             candidate_identity="Not uniquely determined by current data; this report does not propose winners.",
             tradeoff="Smaller cost but narrower optimization claim; all original single permissions still require an explicit treatment of unknown candidates."),
        dict(name="Complete finite-superset pre-screen evidence", action="Only if separately authorized: audit missing raw clean states and exact missing layout/SNR pairs.",
             conditional_clean_calls=reports["ALL"]["conditional_cal200_new_clean_count"],
             missing_bler_points=reports["ALL"]["missing_exact_layout_snr_points"],
             packet_cost_formula="1852 * B new body packet trials for a separately frozen per-point sample count B; no B chosen here.",
             illustrative_not_authorized_B256_packet_trials=1852*256,
             tradeoff="Large additional R0 budget; not hidden inside the registered R1 36000 physical frames.")],
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), new_clean_calls=0, new_bler_trials=0,
    forbidden_inference="No measured performance or ranking is assigned to missing probabilities; no new failure thresholds or policies are chosen.")
with (D / "supplement_design.json").open("x", encoding="utf-8") as f: json.dump(value,f,sort_keys=True,indent=2)
with (D / "supplement_design.py").open("x", encoding="utf-8") as f: f.write(Path(__file__).read_text())
text = """# R0 coverage and supplementary-budget design

This is a metadata audit and design, not an execution authorization or shortlist.

| Family | Profiles | Endpoints | Positive-K endpoints | Unique body layouts | Known layout/SNR pairs | Missing pairs |
|---|---:|---:|---:|---:|---:|---:|
"""
for name,r in reports.items():
    text += f"| {name} | {r['profile_count']} | {r['endpoint_count']} | {r['positiveK_count']} | {r['unique_group_layouts']} | {r['known_exact_layout_snr_points']} | {r['missing_exact_layout_snr_points']} |\n"
text += """
The union has 44 unique endpoints: 8 complete-scale and 36 partial-scale.
The archived clean table has complete scales m=4..8 and gray (under receiver
`gray`, separately from `VAR`); m=1..3 are absent. Positive-K values in that
archive use entropy ordering and cannot represent raster partial tokens.
Conditional on matching the old model/metric/source identities, completing all
current clean-table gaps on cal200 would require 39 * 200 = 7,800 clean VAR
reconstructions and DINO scores. A separately audited raster cache could reduce
this count. For the U/EEP endpoint set alone, 9 * 200 = 1,800 are missing.

Only U profiles 531 and 573 have both group BLER curves present. Their raw-clean
endpoints (m7,K25) and (m6,K48) would cost 400 cal200 clean reconstructions;
including the two positive-K fixed anchors raises this illustrative minimum to
800. This would only evaluate that restricted diagnostic, excluding 199 U
alternatives and every q6 single alternative from comparable screening. It is
not a fair or uniquely justified full-domain nomination.

The full union lacks 1,852 exact layout/SNR points across true SNR 4/7/10/13.
No interpolation, step approximation, or probabilities of zero or one were
inserted. At an illustrative 256 trials per point this would be 474,112 new body
packet trials, but no new sample count or packet budget is authorized here.

The old table describes random scrambled payloads and conditional group
independence, and explicitly leaves image-payload validation incomplete. Its
clean DROP mixture is only a pre-screen approximation; it also does not fully
describe KEEP reconstruction from CRC-rejected actual hard bits. The registered
actual R1 measurements remain necessary. All 689 candidate permissions remain
in the catalogue; no R1 winner or shortlist is selected by this audit.
"""
with (D / "SUPPLEMENT_DESIGN.md").open("x",encoding="utf-8") as f: f.write(text)
print(json.dumps({name:{k:v for k,v in r.items() if not isinstance(v,list)} for name,r in reports.items()},sort_keys=True))
