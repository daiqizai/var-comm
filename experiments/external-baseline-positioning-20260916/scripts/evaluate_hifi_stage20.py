#!/usr/bin/env python3
"""Run the frozen author evaluator on a registered20-source/one-noise cohort."""

import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from var_comm.study import sha256, write_json


class OneRegisteredNoise(ast.NodeTransformer):
    def __init__(self):
        self.replacements = 0

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "seeds":
            self.replacements += 1
            return ast.copy_location(ast.Assign(targets=node.targets, value=ast.List(elts=[ast.Constant(value=2001)], ctx=ast.Load())), node)
        return self.generic_visit(node)


def run(arguments):
    root, stage = arguments.root, arguments.stage
    protocol = json.loads((stage / "protocol.json").read_text())
    if protocol["noise_seeds"] != [2001] or protocol["stage_inference_variant"] != "original_forced_attention_checkpoint":
        raise RuntimeError("only the registered original-execution stage is allowed")
    output = stage / "inference_001"
    if (output / "completion.json").exists():
        raise RuntimeError("completed exploratory inference must not be repeated")
    output.mkdir(exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    evaluator = EXPERIMENT / "scripts/evaluate_authors.py"
    originals = json.loads((root / "author_queue_001/bindings.json").read_text())
    if any(sha256(path) != expected for path, expected in originals.items()):
        raise RuntimeError("original author implementation changed")
    targets = {f"rate2_seed2001_snr{snr:g}_source{index:04d}" for index in protocol["source_indices"] for snr in protocol["snrs_db"]}
    reused = []
    for source in sorted((root / "development_hifi_001/frames").glob("*/frame.json")):
        saved = json.loads(source.read_text())
        key = saved["frame_key"]
        if key not in targets:
            continue
        archive_paths = {Path(row["image_archive"]).resolve() for row in saved["rows"]}
        if len(archive_paths) != 1:
            raise RuntimeError("inconsistent committed frame archive")
        archive = next(iter(archive_paths))
        if not archive.is_relative_to((root / "development_hifi_001/frames").resolve()) or sha256(archive) != saved["archive_sha256"]:
            raise RuntimeError("existing reusable frame is not an intact original artifact")
        destination = output / "frames" / key
        destination.mkdir(exist_ok=True)
        cached = {**saved, "reuse_source_frame_json": str(source), "reuse_source_receipt_sha256": sha256(source),
                  "reuse_scope": "actual_original_frame_same_source_SNR_noise_observation;not_GT_repair"}
        if (destination / "frame.json").exists():
            if json.loads((destination / "frame.json").read_text()) != cached:
                raise RuntimeError("reused frame was already committed differently")
        else:
            write_json(destination / "frame.json", cached)
        reused.append({"frame_key": key, "source_receipt": str(source), "source_receipt_sha256": sha256(source),
                       "image_archive": str(archive), "archive_sha256": saved["archive_sha256"]})
    write_json(stage / "reuse_manifest.json", {"frames": reused, "images_copied_or_linked": False,
               "protocol_sha256": sha256(stage / "protocol.json"), "reused_frames": len(reused)})
    module = ast.parse(evaluator.read_text())
    transformer = OneRegisteredNoise()
    module = transformer.visit(module)
    if transformer.replacements != 1:
        raise RuntimeError("author driver seed assignment is not the reviewed one")
    ast.fix_missing_locations(module)
    namespace = {"__name__": "frozen_author_stage20", "__file__": str(evaluator)}
    exec(compile(module, str(evaluator), "exec"), namespace)
    original_sha = namespace["sha256"]

    def referenced_archive_checksum(path):
        path = Path(path)
        if path.name == "reconstructions.npz" and not path.exists() and path.parent.is_relative_to(output / "frames"):
            saved = json.loads((path.parent / "frame.json").read_text())
            if "reuse_source_frame_json" not in saved:
                raise RuntimeError("new frame has a missing actual archive")
            source = Path(saved["reuse_source_frame_json"])
            if sha256(source) != saved["reuse_source_receipt_sha256"]:
                raise RuntimeError("reused original frame receipt changed")
            return original_sha(Path(saved["rows"][0]["image_archive"]))
        return original_sha(path)

    namespace["sha256"] = referenced_archive_checksum
    metadata = {"scope": "exploratory20_sources_times5SNR_times_one_original_noise", "original_1500_completed": False,
                "expected_frames": 100, "reused_frames": len(reused), "new_frames_expected": 100 - len(reused),
                "forward_and_sampling_code": "unchanged_original_evaluator_and_adapters;only_noise_list_and_manifest_scope_restricted",
                "in_memory_driver_changes": ["seeds=[2001]", "checksum_existing_archive_at_its_original_path"],
                "no_files_copied_or_symlinked_from_original_image_archives": True,
                "bindings": {str(evaluator): sha256(evaluator), str(Path(__file__)): sha256(Path(__file__)),
                             str(stage / "protocol.json"): sha256(stage / "protocol.json"),
                             str(stage / "development_inputs.json"): sha256(stage / "development_inputs.json")}}
    if (stage / "inference_scope.json").exists() and json.loads((stage / "inference_scope.json").read_text()) != metadata:
        raise RuntimeError("resumed exploratory scope changed")
    write_json(stage / "inference_scope.json", metadata)
    options = types.SimpleNamespace(family="hifi", inputs=stage, output=output, calibration=False, limit=None,
                                    paired_adjscc=root / "development_adjscc_001/per_frame.csv")
    namespace["run"](options)
    receipt = json.loads((output / "completion.json").read_text())
    if receipt["frames"] != 100 or receipt["rows"] != 300 or receipt["sources"] != 20:
        raise RuntimeError("exploratory inference has not covered the exact registered cohort")
    write_json(stage / "inference_completed.json", {"status": "STAGE20_ORIGINAL_AUTHOR_INFERENCE_COMPLETE",
               "frames": 100, "rows": 300, "source_images": 20, "original_1500_completed": False,
               "inference_receipt_sha256": sha256(output / "completion.json"), "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    run(parser.parse_args())
