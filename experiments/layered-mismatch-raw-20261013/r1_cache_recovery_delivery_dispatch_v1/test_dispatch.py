"""Synthetic dispatch control tests; no scientific inputs or processes."""
import contextlib
import ast
import base64
import zlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('dispatch_fixture_module',Path(__file__).with_name('dispatch_once.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
REAL_LOAD_ENTRY=d.load_entry

class FakeChild:
    pid=31415
    def __init__(self,outcomes=(0,),poll=None):
        self.outcomes=list(outcomes);self.poll_result=poll;self.timeouts=[];self.terminated=0;self.killed=0;self.fd=None
    def wait(self,timeout=None):
        self.timeouts.append(timeout)
        os.fstat(self.fd)  # Parent lock remains open until each real wait returns.
        result=self.outcomes.pop(0) if len(self.outcomes)>1 else self.outcomes[0]
        if isinstance(result,BaseException):raise result
        return result
    def poll(self):return self.poll_result
    def terminate(self):self.terminated+=1
    def kill(self):self.killed+=1

class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'code';self.tmp=self.base/'tmp';self.root.mkdir();self.tmp.mkdir()
        self.entry=self.root/'experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_v1/deliver.py'
        self.entry.parent.mkdir(parents=True);self.entry.write_text('# synthetic delivery source\n')
        self.control=self.tmp/'r1_cache_recovery_delivery_dispatch_v1';self.child=FakeChild();self.spawn=[];self.closure_calls=[];self.lock_calls=[]
        def preflight(original,root,temporary,reg):
            self.assertIs(original,self.original)
            self.assertFalse(self.control.exists());self.assertEqual(root,self.root);self.assertEqual(temporary,self.tmp)
            self.assertEqual(reg['sha256'],d.MASTER);self.closure_calls.append(reg)
        self.preflight=mock.Mock(side_effect=preflight)
        self.original=object()
        self.module=types.SimpleNamespace(preflight=self.preflight,load_original=mock.Mock(return_value=self.original))
        self.ident=lambda pid:{'pid':pid,'start_ticks':1000+pid}
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(d,'ENTRY_SHA',hashlib.sha256(self.entry.read_bytes()).hexdigest()))
        self.stack.enter_context(mock.patch.object(d,'load_entry',return_value=self.module))
        self.stack.enter_context(mock.patch.object(sys,'platform','linux'))
        self.stack.enter_context(mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}))
        fake_fcntl=types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda fd,flags:self.lock_calls.append((fd,flags)))
        self.stack.enter_context(mock.patch.dict(sys.modules,{'fcntl':fake_fcntl}))
    def spawn_child(self,argv,**kwargs):
        self.assertEqual(len(self.closure_calls),1)
        self.spawn.append((argv,kwargs));self.child.fd=kwargs['pass_fds'][0]
        kwargs['stdout'].write('synthetic child output\n');kwargs['stderr'].write('synthetic stderr\n')
        return self.child
    def run_case(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return d.run(self.root,self.tmp,python='fixture-python',popen=self.spawn_child,identity_fn=self.ident)
    def read(self,name):return json.loads((self.control/name).read_text())
    def test_success_exact_pins_argv_identity_and_lock(self):
        self.run_case();launch=self.read('launch.json');wait=self.read('actual_wait.json')
        expected=['fixture-python','-B',str(self.entry),'--expected-registration',d.MASTER]
        self.assertEqual(launch['argv'],expected);self.assertEqual(self.spawn[0][0],expected)
        self.assertEqual(launch['child'],{'pid':31415,'start_ticks':32415});self.assertEqual(wait['child'],launch['child'])
        self.assertEqual(launch['delivery_entry']['sha256'],hashlib.sha256(self.entry.read_bytes()).hexdigest())
        self.assertTrue(wait['actual_wait']);self.assertEqual(wait['returncode'],0);self.assertEqual(wait['scientific_calls'],0)
        self.assertEqual(self.child.timeouts,[7200]);self.assertFalse((self.control/'failure.json').exists())
        self.assertEqual(self.spawn[0][1]['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(self.lock_calls[0][1],6)
        self.assertEqual(self.spawn[0][1]['cwd'],self.root)
        for name in ('stdout','stderr'):
            value=wait[name];self.assertEqual(value['sha256'],hashlib.sha256(Path(value['path']).read_bytes()).hexdigest())
        with self.assertRaises(OSError):os.fstat(self.child.fd)
    def test_preflight_failure_creates_no_control_or_child(self):
        self.preflight.side_effect=RuntimeError('not closed')
        with self.assertRaisesRegex(RuntimeError,'not closed'):self.run_case()
        self.assertFalse(self.control.exists());self.assertFalse(self.spawn)
    def test_existing_paths_refused_before_preflight(self):
        paths=(self.control,self.tmp/'r1_cache_recovery_delivery_v1',self.tmp/'r1_cache_recovery_delivery_v1.claim.json',
               self.tmp/'r1_cache_recovery_delivery_v1_publication_failure.json',self.root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1')
        for path in paths:
            with self.subTest(path=path):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('existing proof')
                with self.assertRaises(FileExistsError):self.run_case()
                self.assertEqual(path.read_text(),'existing proof');self.assertFalse(self.spawn);self.preflight.assert_not_called()
                path.unlink()
    def test_source_change_during_preflight_stops_before_control(self):
        def changed(*args):self.entry.write_text('changed')
        self.preflight.side_effect=changed
        with self.assertRaisesRegex(RuntimeError,'changed after'):self.run_case()
        self.assertFalse(self.control.exists());self.assertFalse(self.spawn)
    def test_load_entry_rejects_wrong_source_hash(self):
        self.entry.write_text('changed before import')
        with mock.patch.object(d,'load_entry',side_effect=REAL_LOAD_ENTRY):
            with self.assertRaisesRegex(RuntimeError,'source changed'):self.run_case()
        self.assertFalse(self.control.exists());self.assertFalse(self.spawn)
    def test_cuda_not_hidden_refused_before_control(self):
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(RuntimeError,'CUDA hidden'):self.run_case()
        self.assertFalse(self.control.exists());self.assertFalse(self.spawn)
    def test_nonzero_child_retained_no_retry(self):
        self.child=FakeChild((7,))
        with self.assertRaisesRegex(RuntimeError,'CPU delivery failed'):self.run_case()
        self.assertTrue(self.read('actual_wait.json')['actual_wait']);self.assertEqual(self.read('failure.json')['returncode'],7)
        self.assertIs(self.read('failure.json')['automatic_retry'],False)
        before={p.name:p.read_bytes() for p in self.control.iterdir()}
        with self.assertRaises(FileExistsError):self.run_case()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.control.iterdir()});self.assertEqual(len(self.spawn),1)
    def test_timeout_then_terminate_preserves_actual_wait(self):
        self.child=FakeChild((subprocess.TimeoutExpired('fixture',7200),-15))
        with self.assertRaisesRegex(RuntimeError,'CPU delivery failed'):self.run_case()
        wait=self.read('actual_wait.json');self.assertTrue(wait['actual_wait']);self.assertTrue(wait['timed_out']);self.assertTrue(wait['terminated'])
        self.assertFalse(wait['killed']);self.assertEqual(self.child.timeouts,[7200,15]);self.assertEqual(self.child.terminated,1)
    def test_timeout_kill_preserves_actual_wait(self):
        self.child=FakeChild((subprocess.TimeoutExpired('fixture',7200),subprocess.TimeoutExpired('fixture',15),-9))
        with self.assertRaisesRegex(RuntimeError,'CPU delivery failed'):self.run_case()
        wait=self.read('actual_wait.json');self.assertTrue(wait['actual_wait']);self.assertTrue(wait['killed'])
        self.assertEqual(self.child.timeouts,[7200,15,None]);self.assertEqual(self.child.killed,1)
    def test_popen_failure_is_not_actual_wait(self):
        with mock.patch.object(self,'spawn_child',side_effect=OSError('launch unavailable')):
            with self.assertRaisesRegex(RuntimeError,'CPU delivery failed'):self.run_case()
        self.assertFalse((self.control/'actual_wait.json').exists());self.assertIs(self.read('failure.json')['actual_wait'],False)
    def test_child_identity_failure_still_waits_but_fails(self):
        self.ident=lambda pid:{'pid':pid,'start_ticks':True} if pid==31415 else {'pid':pid,'start_ticks':1000+pid}
        with self.assertRaisesRegex(RuntimeError,'CPU delivery failed'):self.run_case()
        wait=self.read('actual_wait.json');self.assertIs(wait['actual_wait'],True);self.assertIsNone(wait['child'])
        self.assertIn('integer PID',wait['metadata_error']);self.assertTrue((self.control/'failure.json').exists())
    def test_unexpected_wait_failure_never_claims_actual_wait(self):
        self.child=FakeChild((OSError('wait failed'),))
        with self.assertRaisesRegex(OSError,'wait failed'):self.run_case()
        self.assertIs(self.read('actual_wait.json')['actual_wait'],False);self.assertIs(self.read('failure.json')['actual_wait'],False)
    def test_bool_wait_code_refused(self):
        self.child=FakeChild((False,))
        with self.assertRaisesRegex(RuntimeError,'integer exit code'):self.run_case()
        self.assertIs(self.read('actual_wait.json')['actual_wait'],False);self.assertTrue((self.control/'failure.json').exists())
    def test_strict_identity_types_and_values(self):
        for value,pid in (({'pid':True,'start_ticks':10},1),({'pid':12,'start_ticks':False},12),({'pid':12,'start_ticks':0},12),({'pid':13,'start_ticks':10},12)):
            with self.subTest(value=value),self.assertRaises(RuntimeError):d.checked_identity(value,pid)
        self.assertEqual(d.checked_identity({'pid':12,'start_ticks':10,'state':'R'},12),{'pid':12,'start_ticks':10})

    def test_registration_schema_and_source_are_specific_to_recovered_delivery(self):
        self.assertEqual(d.MASTER,'ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c')
        self.assertEqual(d.ENTRY_SHA,hashlib.sha256(self.entry.read_bytes()).hexdigest())  # Fixture replaces only source hash.
        self.assertEqual(d.SCHEMA,'R1_RECOVERED_CPU_DELIVERY_DISPATCH_V1')
        self.run_case()
        self.module.load_original.assert_called_once_with(self.root)
        self.assertIn('/r1_cache_recovery_v1/registered/registration.json',self.read('actual_wait.json')['registration']['path'].replace('\\','/'))

# Original 3f32 dispatcher bytes are test data only; never executed.
ORIGINAL_DISPATCH_SOURCE_B85 = (
    'c-qxjTXUm25`OouU}qnIIs{{9a&wNhD&shn)N$gJ<GJnZlu8C+Yr}03B=)KK@!Kr{hQygO`?#(O2qAU1`ua6__wL=5NJPPnH1YXC'
    'nnYV;eE-o1cqE>9zBMwI^Q1U8q%YtuLg5>J8fOtNM4A{v8qCOYp7Z~dLIN50-Me?aFi&HH(y%PboKtFuI7{=wU`dh|4EM=iJ(aJ3'
    'f#i;{W^x^grBf}2p<5Ora*Th=G;xenI))Uhgu#p~msy_rT;kjoE`^A>W}2}Asni_r;j!176TDkyKTe8%7E1g(*!sh(_p^SS!bumL'
    '#eE+}{YaEisfV8~=5#!}y*2IL^p^+#U!F5@P{dh9M_mt|SbhL1g9U60iVA2fWXuYGO>_24hu-jb;0+Fe&G`N+`E?<IEJd<3FB4Qi'
    'P08!?<Vp5H9e?tKMnUG&Y<oBu^p_&(XWL?(CT8#U;$bnF6Z6DdF1?fYC+y_?89NM)4}*ah9EHL1G-UpOg@-3>$PZ8W>F{vi4ZYLg'
    'aNr-qLl|(6`R`B6-sEmE|4JXOFNk^eetCR$ddvr~ogI6p!_Z^ikR7wL!RZkYJvm#R4UgI38896j9SzuE>7Se)1>W&83<k?Xv-dE*'
    'p4<YZ`GA5G(#y&8=F<eoU*0_2Uo6Jg^wR)k|3V)o<JsNi134LbUJrc04br5b&pCJ?U2Y4GdLoOw<h@?NLqoY|&aEdFmE4KJbJ&X8'
    'qn-gj5gOo@hI%yu`$=xhwMHj-p5|7)T2qp1W}%Z+o7bfBTNhpokB%){O+Slbt(;G<b<>O|mYFY2+hEd=MGm&q1n?K8x#6$N<k1ii'
    'Q3cPfh{el*jr2knE`fFW`|m@~c9y1TkJfIP1+3tf+ND|M9D*$|*6x~b0<q$<uqu%=kyIjAYXadHYK8y<{mMb5NrGcnLX28A+^7_P'
    'az`JGGzYJ#mYRm66j!MujMz$&RJt?zc|N<Fe*K}w<N4%bp@)+%<LQqT2Nu8@g4&$XyV=!jI-Pws9j~Q5SRDpRZ!l7K#Z7T=y!6-u'
    'Wth*V!xONzNLIuwi|};6f-!}-K^bS21L*gTNCHS{WN4pPidOz(ng2@6W-5$Ixwh<c6i03o->T<D#FNyjwIPs_D7IE6f=U;P6|t(!'
    'feFjB`%2-kq;H~@O<UQP7JO4wo>MRp5w-Q3y1ID?Vp^tc+P@7)-(8tSqF&oErwE8h)ZK>jvAm##@E@g`M55mYXWwgS{WbR=ArUoF'
    'YqjiTuj8fIW?YfPL=Ti8g$IV3)WU%wEfe0Gc4vi_U%mKF2;3S=;BOy|mc;vMwHJK<hv#%Cnv)Il3*5q2JU8xdF8gYIW0v|rAn0Dr'
    'wFT~IZy)zciEJr(QnZsnk#9#Tf+8J4YLah!o)yLw_{&{dT%~0asJv1IqZUVH9b7Wm2w6y-cue&Gu5CcD@EC~1r7Vmkhl9$D`vngi'
    'mAZ1CVQC3QhWT3)(Up`@3jPIh=G(TUwSAkQCZx&Zj(Md@#|{F>w9I`DUc-`Aqe7H`bZfPXa}?_^#qgj}>Z@uGnQuUAYBUfsQ`OJU'
    'N-de`C>a!4D0OATUlP`{nlv)8@B%RkR~JvnbC+*`w_a!ITgTLTs;UNMV&!RCkhz0;O>>rS$<)!ckw6u8GE6vRhcn!JoN#<N(Idgq'
    '01|=b(?s}A?PD}dNIib3W++f#q5=|q>?&W9nBy!n5=ptioCS^!83qcs$RSm!kQjfwyr7?M9&SENCr}AL-Hay>CK$>zA!+7fB>)Q<'
    'QzW1OTeUX^0dxYz{l;1ZAks_sS6U~ig1*T&8P5f7lQlp&r~%3W0F;9op!C~Dd>Zttv72qp>aZ(t*IZegbW4oplyg}|J1+ZS5OP6N'
    'mPiKKXtKDW-{#<APB-nkI%zzkPv-*^$kT{4w(altzo&r~9Lq0?iq+SuHal-@hCOe;a{Y*jSWEWyDw>tcNa)C*A!KI2$km?zQBQ5D'
    '>`^-(wL__u0o1kq(go-qMq;%tl#O;mJ8fOthVgf!Nk$+{6d4c-5(GD5DCGb}n1P*x`zpC;j-Bi>!Kpf#7%T+a8jToc$AoVD7>FFh'
    'nG-KP@jQp46f}J#S1giMgjZNU^phe|6Dmq;)KHP?o3-6td~G@kjm$4U&-X%<3SnsfK*9h7V4H;(M^8*=<3C{SkJi)O2fYKjnkVv2'
    '=M%>~_+TR9sB{6(2N-jhz7N`3HXZH6-zyEoaUDkz>UGSB2FD%CkoJSq_=Apb#C;V{B@kY^brXN-bHa1-3K6uj8UirLNKhOB?l_%?'
    'eK`sf)UH{iK6r~H-fPA&+fWHWJqaXfL3*dQ6G(+8Q22@Jf<h55$uxT6z+9^uVQ6Dwa3gG2gDRY~4(pWHq(`mPUAleq6!%{j*C5jD'
    'Zt;13vzUDNx&RR`C&V1o@9t(FA13n$c!`sClTH#AbCpia+3h{O`*=$i*Wh5650JZW|Cqj*ncYwBK1?qjUQEN&<jeiMGi#4p2l<9<'
    'z7un(^O(ODQ9$m}grBP#!vmrU7NA!wSpiMULdVTH#9R#Z1DEKDM-fHM14sCw2`QT375YinQq^0j=VvAw5=sQDt~g26J`33*Q(SEC'
    ';CKSL0-!-;=>5&SM$Brfi?1vjo3n}tVnU>2aA0P%7S&9uyWNoMfLPt7RKU_P$GBhd*PyCG2`p(qBxIVxz@wTwiq&~lU%BI=b}IS}'
    '3f%3$O1yJWSp*=WTWk>e>!xs1w=DR^=`001ys;GCj5-0OZLeU9rQ+Ow4pf1M{^*Ice8$m?NX1XnBKig}OiCI;JnB4W+pl?`FZ?uF'
    'Xv?TJWH;>4>4a>=$^>$<4p<eKI%@T1QuB0tXp}(rv`oZCBQU%)X`ue0@>^co^2Jt)>29e2dJF6=v?onA8Dgf<tIV~FCiH}MFsWyK'
    '>}J-oUlVv!%E8g=)qg2q6}&CxI0Iq7pGT)J`M@MJ9XQ0#dEi|zzdNzc1dvssdTsojXd=|!fboPK2nJ?XPxT5nOHq_MdiN;v|5AGm'
    'bx1~|pDhd#rbn%e&`A%!rc0e>d@i!8cI^)lXn{)>Q51pbOy#sB+YMir;um(<4V_9k`sVB$zZp3^Vwr?-UCOJHb5NC_{miN)g!O%I'
    'Zmjdt%LPqFbZXp5wTr6jYvXrPO-nyD-quRBU)n=aYum$s$DIQIMm(Hg$=0%Vig=|p+aCQ~pf31VmTjHp4dd;xGAoM|Is^b)unvHq'
    '#>Tr-)jkN22Vpx`+1>77_2$ItwAhYn+PjZ6I}tjcXIesnR&WM&_SdR`65ReT1FC`t=b+ZAv8_LO0DUHsrgP15{GQc(&y$u0$OS7('
    'NzBeS<MH*)^b!F2&4T`OGM^zF)O{gh@oH>$)Y<X2SKeapLA6+Zdf+gWA{3((x^qfn5SUU^8(;l*BPaF$IoykURbm~!SMyfDrJoCx'
    'd<msN>I23-*Lz$R1e8_Ft+%^|y0=#utm!}STdty_Zb(O}l%~y%k8Vwmb~=ixm4-fTlR=w`0EuD$Z9wzZ`i^jpA5rYye*yLVzxV'
)

class SourceReuseTests(unittest.TestCase):
    def test_inverse_ast_keeps_original_wait_and_failure_control(self):
        source=Path(d.__file__).read_text(encoding='utf8')
        replacements=(
            ('ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c','70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47'),
            ('f37df9e92520a1f45bdc7954dbbd6eaf5c1a7cab4721d3b5df7ec5df65c6de01','94b6986e1db8a6082f0a02a6a91855fd79b926a390bb1551a1bc785d06bfd1b3'),
            ('R1_RECOVERED_CPU_DELIVERY_DISPATCH_V1','R1_CPU_DELIVERY_DISPATCH_V1'),
            ('r1_cache_recovery_delivery_dispatch_v1','r1_delivery_dispatch_v1'),
            ('r1_cache_recovery_delivery_v1','r1_delivery_v1'),
            ('r1_cache_recovery_v1/registered/registration.json','r1_execution_v1/registered/registration.json'),
            ('    original=module.load_original(root)\n    module.preflight(original,root,temporary,registration)  # Explicit old failed/new successful segments only.',
             '    module.preflight(root,temporary,registration)  # Existing small closure metadata only.'),
        )
        for old,new in replacements:
            self.assertIn(old,source);source=source.replace(old,new)
        original_bytes=zlib.decompress(base64.b85decode(ORIGINAL_DISPATCH_SOURCE_B85))
        self.assertEqual(hashlib.sha256(original_bytes).hexdigest(),'3f32be6e1d63205cf4d71d7e0cddc95d6f314f14628c8cf08c8a79b432a12748')
        self.assertEqual(ast.dump(ast.parse(source),include_attributes=False),
                         ast.dump(ast.parse(original_bytes.decode('utf8')),include_attributes=False))

if __name__=='__main__':unittest.main()
