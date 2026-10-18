"""Only fresh RX, post-exit audit, or zero-packet constructor admission."""
import argparse,json,os,pathlib,sys
from adapter import E,ORIGINAL,adapt,namespace
sys.path.insert(0,str(E/'phy_v1'));sys.path.insert(0,str(E))
import link
from a1_owner import L,desc,put
CODE=E/'phy_rx_environment_v1';ROOT=L/'a2_phy_rx_environment_v1'
def definitions():
    ns=namespace();exec(compile(adapt(ORIGINAL.read_bytes()),str(ORIGINAL)+'::environment_v1','exec'),ns);return ns
def probe():
    target=ROOT/'constructor_probe';target.mkdir();cache=target/'runtime_cache';cache.mkdir();(cache/'tmp').mkdir()
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
    import tempfile;tempfile.tempdir=str(cache/'tmp')
    from a1_native import install_rx_boundary
    rtroot=link.R.parent.parent/'var_comm_runtime_20261010'
    # These are the two exact interpreter/package roots proved by original CPY.
    allowed=[E,link.OLD,link.R/'experiments/wcl-evidence-closure-20261009/scripts',link.R/'experiments/prior-aware-grouped-mcs-v1',
      link.R/'experiments/main-raw64-20261007',link.R/'src',link.R/'outputs/build',
      link.R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',rtroot/'envs/frozen_ldpc_py311',
      rtroot/'envs/frozen_python_base_v1/lib/python3.11',rtroot/'assets/frozen_namespace_v1/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/lib/python3.11/site-packages',target]
    blocked=install_rx_boundary(allowed);calls=[]
    def forbidden(*a,**kw):calls.append('packet');raise AssertionError('No packet in constructor qualification')
    link.Runtime.transmit=link.Runtime.receive=forbidden;link.ep.Backend.encode=link.ep.Backend.decode=forbidden
    original_loader=link.ep.load_exact
    def loader(path,name):
        module=original_loader(path,name)
        if name=='uep_phy':module.Header.transmit=module.Header.receive=forbidden
        return module
    link.ep.load_exact=loader
    rt=link.Runtime();assert not calls and not blocked and not rt.torch.cuda.is_initialized()
    put(target/'completion.json',{'status':'PASS_CONSTRUCTORS_ONLY','calls':calls,'blocked':blocked,'CUDA_initialized':False,'catalogue_sha256':rt.catalogue['catalogue_sha256'],'backend_identity':rt.backend.identity,'profiles':len(rt.profiles),'sources':[desc(CODE/n) for n in ['adapter.py','worker.py']]})
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['probe','rx','audit']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='probe':probe();return
    ns=definitions()
    if a.role=='rx':assert a.expected_registration is None;ns['rx']()
    else:
        reg=ns['REG'];assert desc(reg)['sha256']==a.expected_registration
        ns['audit'](json.loads(reg.read_text()))
if __name__=='__main__':main()
