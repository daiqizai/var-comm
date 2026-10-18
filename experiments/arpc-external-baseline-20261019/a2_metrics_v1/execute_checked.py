"""CLI-only repair before any preparation: preserve the original launcher bytes."""
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import owner
def success(path):
    w=json.loads(pathlib.Path(path).read_text())
    assert type(w['returncode']) is int and w['returncode']==0
    assert not any(w[k] for k in ['timed_out','terminated','killed'])
    assert w['metadata_error'] is None and owner.gone(w['process'])
    return w
# The original launcher imports this receipt validator, which its owner omitted.
# No model, metric, registration, source/counter or launch path is replaced.
owner.success=success
import execute
if __name__=='__main__':execute.main()
