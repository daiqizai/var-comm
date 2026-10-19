"""Original once-only source supervisor with explicit host-adapter child argv."""
import ast,hashlib,pathlib
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-common500-extension-20261019')
SOURCE_SHA='5fff060c8c2cdc9fc6547d68afb02ba2ff2625f0d3b80f16f31598873f33a95b'
OLD="str(E/(a.stage+'_control.py'))"
NEW="str(E/'source_host_v1.py')"
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==SOURCE_SHA
    s=data.decode();assert s.count(OLD)==1 and NEW not in s
    result=s.replace(OLD,NEW,1);assert result.replace(NEW,OLD,1)==s
    # The sole change is the command's script path; timeout/wait/cleanup stay exact.
    ast.parse(result)
    return result
if __name__=='__main__':
    p=E/'execute.py'
    exec(compile(adapt(p.read_bytes()),str(p)+'::authorized_host3781','exec'),{'__name__':'__main__','__file__':str(__file__)})
