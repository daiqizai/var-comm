#!/usr/bin/env python3
"""Freeze the calibration-selected method and ordered source list before opening holdout pixels."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from var_comm.next_scale_prior import preprocess
from var_comm.study import artifact_hashes, create_output, sha256, write_json


def now():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--development-analysis", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--known-content", type=Path, required=True)
    parser.add_argument("--reference-check", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("PLAN ONLY: method and ordered path list are written before any new holdout image is opened")
        return
    config_path = ROOT / "configs/communication_decision_study.json"
    config = json.loads(config_path.read_text())
    policies = json.loads(arguments.policies.read_text())
    development = json.loads(arguments.development_analysis.read_text())
    inventory = json.loads((arguments.catalog / "inventory.json").read_text())
    content = json.loads(arguments.known_content.read_text())
    reference_check = json.loads(arguments.reference_check.read_text())
    if (policies["status"] != "CALIBRATION_POLICIES_FROZEN" or policies["config_sha256"] != sha256(config_path) or
            development["status"] != "FROZEN_COMMUNICATION_POLICY_EVALUATION_COMPLETE" or development["population"] != "development" or
            development["policy_sha256"] != sha256(arguments.policies) or content["status"] != "KNOWN_USED_CONTENT_EXCLUSION_READY" or
            content["new_holdout_files_opened"] or inventory["image_files_opened"] or
            reference_check["status"] != "FROZEN_REFERENCE_DEVELOPMENT_REPLAY_PASS"):
        raise RuntimeError("finish calibration/development and qualify fixed references before freezing holdout")
    if content["catalog_inventory_sha256"] != sha256(arguments.catalog / "inventory.json"):
        raise RuntimeError("known-content exclusions do not match the source-use inventory")
    catalog = json.loads((arguments.catalog / "candidate_paths.json").read_text())
    if catalog["role"] != "metadata_only_unused_candidates_not_accessed":
        raise RuntimeError("candidate list is not an unaccessed metadata-only catalog")
    groups = {label: [] for label in range(1000)}
    used_indices = set(inventory["used_val_indices"])
    for row in catalog["images"]:
        if row["val_index"] in used_indices:
            raise RuntimeError("candidate catalog contains a previously used source")
        groups[int(row["class_index"])].append(row)
    salt = "VAR_COMM_one_shot_holdout_20260915_v1"
    for label, values in groups.items():
        if not values:
            raise RuntimeError(f"no unused candidate for class {label}")
        values.sort(key=lambda row: hashlib.sha256(f"{salt}|{row['image_id']}".encode()).hexdigest())
    output = create_output(arguments.output_dir)
    selected_method = policies["global_selected_family"] + "_quality"
    equivalences = []
    for family, rules in policies["actions"].items():
        for rule in ("reliability", "goodput"):
            if rules["quality"] == rules[rule]:
                equivalences.append([family + "_quality", family + "_" + rule])
    core = {"status": "METHOD_CORE_FROZEN_BEFORE_HOLDOUT_PIXELS", "frozen_local": now(), "selected_method": selected_method,
        "selection_origin": "the preregistered calibration-primary family choice, not development or holdout reselection",
        "config_sha256": sha256(config_path), "policies_sha256": sha256(arguments.policies), "policies_path": str(arguments.policies.resolve()),
        "development_analysis_sha256": sha256(arguments.development_analysis), "actions": policies["actions"],
        "primary_metric": "LPIPS", "primary_snrs_db": config["primary_snrs_db"], "high_snrs_db": config["high_snrs_db"],
        "transition_snrs_db_secondary": [5., 6.], "all_snrs_db": config["snrs_db"], "noise_seeds": config["holdout_seeds"],
        "source_images": 1000, "PSNR_allowed_drop_db": config["quality_max_PSNR_drop_db"],
        "primary_mechanism_controls": [policies["global_selected_family"] + "_reliability", policies["global_selected_family"] + "_goodput"],
        "mechanism_claim_requires": "negative LPIPS paired CI against both ordinary same-family controls, with PSNR lower CI >= -0.25dB",
        "structurally_identical_policies_before_holdout": equivalences,
        "all_fixed_mode_and_other_policy_controls_retained": True,
        "system_references": ["old_raw_adaptive", "raw_m8", "r2__full_grid_innovation", "r3__full_grid_prediction_features", "perceptual_deepjscc", "wetok_8PSK_FEC"],
        "complex_uses": 3060, "total_energy": 6120., "failure_rules_from_frozen_config": True,
        "TX_true_class_known": True, "nominal_SNR_ideally_shared": True, "DINO_participated_in_old_digital_development": True,
        "DINO_used_for_new_selection": False, "processing_time_excludes_airtime_and_queueing": True,
        "no_method_changes_after_holdout": True, "no_new_training_or_search": True}
    core_path = output / "method_core.json"
    plan_path = output / "ordered_source_plan.json"
    write_json(core_path, core)
    write_json(plan_path, {"status": "ORDERED_SOURCE_LIST_FROZEN_BEFORE_PIXELS", "frozen_local": now(), "role": "one_per_class_in_distribution_holdout",
        "sampling_salt": salt, "selection_rule": "first readable, nonduplicate file/pixel source in frozen per-class hash order",
        "replacement_rule": "unreadable files or exact prior/within-holdout file/pixel duplicates only; never quality or model outputs",
        "catalog_inventory_sha256": sha256(arguments.catalog / "inventory.json"), "known_content_sha256": sha256(arguments.known_content),
        "ordered_candidates": groups})
    pre_pixel = {"status": "METHOD_AND_ORDERED_DATA_LIST_FROZEN", "frozen_local": now(), "holdout_images_opened_so_far": 0,
        "method_core_sha256": sha256(core_path), "ordered_source_plan_sha256": sha256(plan_path)}
    write_json(output / "pre_pixel_freeze.json", pre_pixel)
    if torch.cuda.is_initialized():
        raise RuntimeError("holdout source preparation must not initialize model/GPU inference")
    torch.set_num_threads(4)
    file_hashes, pixel_hashes = set(content["file_hashes"]), set(content["pixel_hashes"])
    accepted, inspections, rejected = [], [], []
    root = ROOT.parent / "VAR-MAP-GATE0/data/imagenet/val"
    first_access = now()
    try:
        for label in range(1000):
            chosen = None
            for candidate in groups[label]:
                path = Path(candidate["path"]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise RuntimeError("holdout candidate escaped the registered ImageNet validation root")
                entry = {"image_id": candidate["image_id"], "class_index": label, "path": str(path)}
                try:
                    file_hash = sha256(path)
                    if file_hash in file_hashes:
                        rejected.append({**entry, "reason": "exact_file_duplicate", "file_sha256": file_hash})
                        continue
                    unused_normalized, pixel_hash = preprocess(path)
                except (OSError, ValueError) as error:
                    rejected.append({**entry, "reason": "unreadable_source", "error": repr(error)})
                    continue
                inspections.append({**entry, "file_sha256": file_hash, "preprocessed_rgb_sha256": pixel_hash})
                if pixel_hash in pixel_hashes:
                    rejected.append({**entry, "reason": "exact_preprocessed_pixel_duplicate", "preprocessed_rgb_sha256": pixel_hash})
                    continue
                chosen = {**candidate, "file_sha256": file_hash, "preprocessed_rgb_sha256": pixel_hash}
                file_hashes.add(file_hash)
                pixel_hashes.add(pixel_hash)
                break
            if chosen is None:
                raise RuntimeError(f"frozen reserve list exhausted for class {label}; no replacement based on performance is permitted")
            accepted.append(chosen)
        if len(accepted) != 1000 or sha256(core_path) != pre_pixel["method_core_sha256"] or sha256(plan_path) != pre_pixel["ordered_source_plan_sha256"]:
            raise RuntimeError("method or ordered source selection changed after pixel access")
        manifest_path = output / "holdout_manifest.json"
        write_json(manifest_path, {"role": "independent_holdout_after_method_freeze", "created_local": now(),
            "method_core_sha256": sha256(core_path), "pre_pixel_freeze_sha256": sha256(output / "pre_pixel_freeze.json"),
            "source_images": 1000, "classes": 1000, "images": accepted})
        write_json(output / "source_integrity_checks.json", {"first_new_holdout_content_access_local": first_access,
            "new_images_decoded_for_integrity": len(inspections), "model_inference_performed": False,
            "inspected": inspections, "rejections": rejected, "quality_based_replacements": 0,
            "coverage_caveat": content["coverage_caveat"]})
        final_method = {**core, "status": "FINAL_METHOD_AND_HOLDOUT_PROTOCOL_FROZEN", "sealed_local": now(),
            "method_core_sha256": sha256(core_path), "holdout_manifest_sha256": sha256(manifest_path),
            "pre_pixel_freeze_sha256": sha256(output / "pre_pixel_freeze.json")}
        write_json(output / "frozen_method.json", final_method)
        write_json(output / "completion.json", {"status": "INDEPENDENT_HOLDOUT_READY_NO_MODEL_INFERENCE", "completed_local": now(),
            "source_images": 1000, "selected_method_from_calibration": selected_method,
            "model_inference_performed": False, "method_frozen_before_any_new_pixels": True,
            "source_script_sha256": sha256(Path(__file__)), "research_goal_complete": False, "output_hashes": artifact_hashes(output)})
        print(json.dumps({"status": "INDEPENDENT_HOLDOUT_READY_NO_MODEL_INFERENCE", "sources": len(accepted),
                          "rejections": len(rejected), "selected_method": selected_method, "equivalent_policies": equivalences}, indent=2))
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error), "first_new_content_access": first_access,
            "accepted_sources": len(accepted), "inspections": inspections, "rejections": rejected})
        raise


if __name__ == "__main__":
    main()
