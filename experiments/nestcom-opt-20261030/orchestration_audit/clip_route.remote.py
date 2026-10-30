from pathlib import Path
import json
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
p=R/'results/nestcom_evidence_20261029_v1/clip_metrics_path_v2_control_registration.json';v=json.loads(p.read_bytes());print(v)
