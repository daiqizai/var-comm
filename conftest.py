"""Assert that tests use this checkout's self-written modules."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SELF_PACKAGES = {"var_comm", "cadsd_jscc", "latent_enhancement", "latent_enhancement_b", "latent_enhancement_eval", "latent_followup", "latent_mechanisms", "latent_enhancement_timing", "wetok_comm", "joint_sender"}

def pytest_sessionfinish(session, exitstatus):
    foreign = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and name.split(".")[0] in SELF_PACKAGES:
            path = Path(file).resolve()
            if not path.is_relative_to(ROOT) or "publish" in path.relative_to(ROOT).parts:
                foreign.append((name, str(path)))
    if foreign:
        session.exitstatus = 1
        print("Foreign self-written imports:", foreign)
