"""Required repository engineering checks; not scientific quality evidence."""
import subprocess
from runtime import R,PY
for name in ['verify_repository.py','run_cpu_checks.py']:
    result=subprocess.run([PY,str(R/'tools'/name)],cwd=R,check=False)
    assert result.returncode==0,name
