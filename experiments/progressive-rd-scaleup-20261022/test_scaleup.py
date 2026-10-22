"""Synthetic CPU qualification of new selection/objective/control; no real assets."""
import unittest,torch,tempfile,pathlib,ast,hashlib,json
from rdscale_policy import *
from rdscale_common import Ledger,definitions
from decoder_model import ProfileDecoder

class ScaleupTests(unittest.TestCase):
    def test_validation_fixed_identity_order(self):
        ids=['source_%04d'%i for i in range(1000)];excluded=ids[::10]
        chosen=choose_validation(ids,excluded)
        self.assertEqual(len(chosen),200);self.assertEqual(len(set(chosen)),200)
        self.assertTrue(set(ids[i]for i in chosen).isdisjoint(excluded))
        expected=sorted([i for i in range(1000)if ids[i]not in excluded],key=lambda i:hashlib.sha256(('progressive-rd-scaleup-validation-v1|'+ids[i]).encode()).hexdigest())[:200]
        self.assertEqual(chosen,expected)
    def test_selection_rejects_duplicate_ids(self):
        with self.assertRaises(AssertionError):choose_validation(['a']*1000,[])
    def test_selection_rejects_unknown_exclusion(self):
        with self.assertRaises(AssertionError):choose_validation([str(i)for i in range(1000)],['unknown'])
    def test_selection_requires_full_count(self):
        with self.assertRaises(AssertionError):choose_validation(['a','b'],['a'],2)
    def test_no_score_in_selection_signature(self):
        import inspect
        self.assertEqual(list(inspect.signature(choose_validation).parameters),['ids','excluded','count'])
    def test_job_identity_unique(self):
        self.assertEqual(len({job_name(a,s)for a in ARMS for s in SEEDS}),6)
    def test_unmodified_prefix_loss(self):
        a=torch.tensor(.4,requires_grad=True);b=torch.tensor(.2,requires_grad=True)
        loss,anchor=objective(a,b);self.assertTrue(torch.equal(loss,a+.1*b));self.assertIsNone(anchor)
        loss.backward();self.assertEqual(float(a.grad),1.);self.assertAlmostEqual(float(b.grad),.1,places=7)
    def test_anchor_preserves_teacher_no_gradient(self):
        a=torch.tensor(.1,requires_grad=True);b=torch.tensor(.2,requires_grad=True)
        student=torch.tensor([2.,4.],requires_grad=True);teacher=torch.tensor([1.,1.],requires_grad=True)
        loss,anchor=objective(a,b,student,teacher);self.assertEqual(float(anchor),5.)
        loss.backward();self.assertIsNone(teacher.grad);self.assertTrue(torch.equal(student.grad,torch.tensor([1.,3.])))
    def test_equal_full_outputs_add_zero(self):
        x=torch.randn(2,3,4,4);a=torch.tensor(.25);b=torch.tensor(.3)
        loss,anchor=objective(a,b,x,x.clone());self.assertEqual(float(anchor),0.);self.assertTrue(torch.equal(loss,a+.1*b))
    def test_teacher_required(self):
        with self.assertRaises(AssertionError):objective(torch.tensor(1.),torch.tensor(1.),torch.ones(1),None)
    def test_teacher_without_student_rejected(self):
        with self.assertRaises(AssertionError):objective(torch.tensor(1.),torch.tensor(1.),None,torch.ones(1))
    def test_capacity_and_initial_output_match(self):
        x=torch.randn(2,32,2,2);ep=torch.tensor([0,7]);models=[ProfileDecoder(torch.nn.Identity(),False)for _ in ARMS]
        for m in models:
            self.assertTrue(torch.equal(m(x,ep),x));self.assertEqual(sum(v.numel()for v in m.parameters()),576)
    def test_ledger_no_duplicate_after_success(self):
        with tempfile.TemporaryDirectory()as d:
            l=Ledger(pathlib.Path(d)/'ledger.jsonl');self.assertEqual(l.call('x','1',lambda:2),2)
            with self.assertRaises(AssertionError):l.call('x','1',lambda:3)
    def test_ledger_no_duplicate_after_failure(self):
        with tempfile.TemporaryDirectory()as d:
            l=Ledger(pathlib.Path(d)/'ledger.jsonl')
            with self.assertRaises(ValueError):l.call('x','1',lambda:(_ for _ in ()).throw(ValueError('fixture')))
            with self.assertRaises(AssertionError):l.call('x','1',lambda:2)
            self.assertEqual([json.loads(v)['status']for v in (pathlib.Path(d)/'ledger.jsonl').read_text().splitlines()],['STARTED','FAILED'])
    def test_nested_anchor_accounting(self):
        with tempfile.TemporaryDirectory()as d:
            l=Ledger(pathlib.Path(d)/'ledger.jsonl');l.call('update','1',lambda:l.call('teacher','1',lambda:None))
            e=[json.loads(v)for v in(pathlib.Path(d)/'ledger.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['status']=='COMPLETE'for x in e),2)
    def test_native_runtime_mathematics_unchanged(self):
        here=pathlib.Path(__file__).resolve().parent;old=here.parent/'progressive-rd-20261022/seed_runtime_v2.py'
        a=ast.parse((here/'rdscale_runtime.py').read_text());b=ast.parse(old.read_text())
        defs=lambda t:{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(defs(a),defs(b))
    def test_source_receiver_call_arguments_unchanged(self):
        here=pathlib.Path(__file__).resolve().parent
        old=ast.parse((here.parent/'progressive-rd-20261022/seed_worker_v2.py').read_text());new=ast.parse((here/'validation_worker.py').read_text())
        def calls(t):return [ast.dump(n,include_attributes=False)for n in ast.walk(t)if isinstance(n,ast.Call)and((isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='codec')or(isinstance(n.func,ast.Name)and n.func.id=='render'))]
        self.assertEqual(calls(old),calls(new))
    def test_training_shape_and_seed_budget_static(self):
        self.assertEqual(len(ENDPOINTS),8);self.assertEqual(ENDPOINTS[-1],(10,0));self.assertEqual(len(ARMS)*len(SEEDS)*10000,60000)
        self.assertEqual(6*200*8,9600)
    def test_order_state_replay_synthetic(self):
        here=pathlib.Path(__file__).resolve().parent;root=here.parents[1]
        ns=dict(torch=torch);definitions(root/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/training.py',{'PairedOrder'},ns)
        a=ns['PairedOrder'](48,7);b=ns['PairedOrder'](48,7)
        self.assertTrue(torch.equal(a.next(17),b.next(17)));state=a.state_dict();expect=a.next(40);b.load_state_dict(state);self.assertTrue(torch.equal(expect,b.next(40)))
    def test_no_cuda_initialization(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
