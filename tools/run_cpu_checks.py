#!/usr/bin/env python3
"""Run selected asset-free CPU regressions using only this checkout's self-written code."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/var-latent-enhancement-20260917"
SOURCE_DIRS = [ROOT / "src", EXP / "src", *[EXP / n / "src" for n in ("phase_b", "evaluation", "followup", "mechanisms", "timing", "research")]]

SOURCE_DIRS.append(ROOT / 'experiments/var-short-prefix-hybrid-20260923/src')
SOURCE_DIRS.append(ROOT / 'experiments/token_channel_efficiency_20260923/src')

def main():
    # Replace inherited PYTHONPATH. Installed third-party dependencies may use
    # user site; conftest rejects self-written modules outside this checkout.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, SOURCE_DIRS)),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    modules = ["var_comm.entropy", "var_comm.hybrid_training", "cadsd_jscc.strong_jscc",
               "cadsd_jscc.exact_budget_strong_jscc", "latent_enhancement.latent",
               "latent_enhancement_b.train", "latent_enhancement_eval.deployment",
               "latent_followup.policy_development", "latent_mechanisms.predictor_innovation", "latent_research.models", "latent_research.train", "latent_research.evaluate",
               "latent_research.digital_requalify", "latent_research.system_policy", "short_prefix.protocol", "short_prefix.models", "short_prefix.train", "short_prefix.execution", "token_efficiency.source", "token_efficiency.phy", "token_efficiency.models", "token_efficiency.budget_train", "token_efficiency.coordinator", "token_efficiency.execution", "token_efficiency.qualify_execution", "token_efficiency.statistics", "token_efficiency.publish_source", "token_efficiency.retired_gate", "token_efficiency.coordinator_v2"]
    probe = """import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
result={}
for name in json.loads(sys.argv[2]):
 m=importlib.import_module(name); p=Path(m.__file__).resolve()
 assert p.is_relative_to(root) and 'publish' not in p.relative_to(root).parts,(name,str(p))
 result[name]=str(p.relative_to(root))
print(json.dumps({'module_files':result,'GPU':'NOT_RUN'},indent=2))
"""
    subprocess.run([sys.executable, "-c", probe, str(ROOT), json.dumps(modules)],cwd=ROOT,env=env,check=True)
    suites = ["tests", *[str((EXP/n).relative_to(ROOT)) for n in
               ("tests", "phase_b/tests", "evaluation/tests", "followup/tests", "mechanisms/tests", "research/tests")]]
    suites.append('experiments/var-short-prefix-hybrid-20260923/tests')
    suites.append('experiments/token_channel_efficiency_20260923/tests')
    return subprocess.run([sys.executable,"-m","pytest","-q",*suites],cwd=ROOT,env=env).returncode

if __name__ == "__main__":
    raise SystemExit(main())
