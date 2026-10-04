"""CPU-only queue tests: original worker contracts, dependency order and cleanup."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# The local review host is Windows; no kernel locking is exercised by these tests.
try:import fcntl
except ImportError:sys.modules.setdefault('fcntl',SimpleNamespace())
import study_owner as owner_module
from uep_common import read,write,sha


class Process:
    def __init__(self,pid,clock,duration,receipt=None,code=0,receipt_value=None):
        self.pid=pid;self.clock=clock;self.end=clock[0]+duration;self.receipt=receipt
        self.code=code;self.returncode=None;self.terminated=False;self.killed=False
        self.receipt_value=receipt_value or dict(status='FIXTURE_COMPLETE')
    def poll(self):
        if self.returncode is None and self.clock[0]>=self.end:
            self.returncode=self.code
            if self.code==0 and self.receipt is not None:write(self.receipt,self.receipt_value)
        return self.returncode
    def terminate(self):self.terminated=True;self.returncode=-15
    def kill(self):self.killed=True;self.returncode=-9
    def wait(self,timeout=None):return self.poll()


class TestOwner(owner_module.Owner):
    __test__=False
    def __init__(self,folder,specifications):
        super().__init__(dict(root=str(folder),out=str(folder),native_python='unused',source_bindings={}))
        self.clock=[0.];self.specifications=specifications;self.starts=[];self.stop_at=None;self.peak=0
    def check(self):
        if self.stop_at is not None and self.clock[0]>=self.stop_at:raise RuntimeError('fixture STOP')
    def start(self,name,args,gpu=False):
        self.check();spec=self.specifications[name]
        if gpu:
            assert not any(p.poll() is None and self.child_records[p.pid]['gpu'] for p in self.children)
        p=Process(100+len(self.children),self.clock,spec['duration'],spec.get('receipt'),spec.get('code',0),spec.get('receipt_value'))
        self.children.append(p);self.child_records[p.pid]=dict(name=name,pid=p.pid,args=args,gpu=gpu)
        self.starts.append((name,self.clock[0],gpu,args));self.record_children()
        self.peak=max(self.peak,sum(p.poll() is None for p in self.children));return p
    def sleep(self,seconds):self.clock[0]+=seconds


class QueueTests(unittest.TestCase):
    def setup_owner(self,folder,cpu_names=('actual_link',),gpu_name='P1024',durations=None,omit_receipts=(),failures=None):
        durations=durations or {};failures=failures or {};specs={};steps=[]
        for name in (*cpu_names,gpu_name):
            receipt=folder/(name+'.json');args=[name+'.py','--same-original-command']
            step=dict(name=name,args=args,gpu=name==gpu_name,receipt=receipt,expected_status='FIXTURE_COMPLETE')
            specs[name]=dict(duration=durations.get(name,4),receipt=None if name in omit_receipts else receipt,
                             code=failures.get(name,0))
            steps.append(step)
        return TestOwner(folder,specs),steps[:-1],steps[-1]

    def run_pair(self,owner,cpu,gpu):
        with patch.object(owner_module.time,'sleep',side_effect=owner.sleep):
            owner.parallel_cpu_gpu('fixture',cpu,gpu)

    def test_actual_cpu_and_P_gpu_start_together_and_keep_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp),durations=dict(actual_link=6,P1024=4))
            self.run_pair(owner,cpu,gpu)
            self.assertEqual([s[:3] for s in owner.starts],[('actual_link',0.,False),('P1024',0.,True)])
            self.assertEqual([s[3] for s in owner.starts],[cpu[0]['args'],gpu['args']])
            self.assertEqual(owner.peak,2);self.assertEqual(owner.clock[0],6.)
            self.assertEqual(read(Path(tmp)/'active_children.json')['children'],[])

    def test_ordered_cpu_chain_while_gpu_evaluation_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp),('model_validation','extension_gate'),'aux_gpu',
                dict(model_validation=2,extension_gate=2,aux_gpu=8))
            self.run_pair(owner,cpu,gpu)
            self.assertEqual([s[:3] for s in owner.starts],
                [('model_validation',0.,False),('aux_gpu',0.,True),('extension_gate',2.,False)])
            self.assertEqual(owner.clock[0],8.);self.assertEqual(owner.peak,2)

    def test_completed_receipts_skip_workers_but_preserve_dependency_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);owner,cpu,gpu=self.setup_owner(folder,('model_validation','extension_gate'),'aux_gpu')
            write(cpu[0]['receipt'],dict(status='FIXTURE_COMPLETE'));write(gpu['receipt'],dict(status='FIXTURE_COMPLETE'))
            self.run_pair(owner,cpu,gpu)
            self.assertEqual([s[0] for s in owner.starts],['extension_gate'])

    def test_worker_failure_terminates_only_other_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp),durations=dict(actual_link=2,P1024=100),failures=dict(actual_link=1))
            external=Process(999,owner.clock,100)
            with self.assertRaisesRegex(RuntimeError,'Parallel worker failed: actual_link'):self.run_pair(owner,cpu,gpu)
            self.assertTrue(owner.children[1].terminated);self.assertFalse(external.terminated)
            self.assertEqual(read(Path(tmp)/'active_children.json')['children'],[])

    def test_exit_without_receipt_blocks_gate_and_stops_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp),('model_validation','extension_gate'),'aux_gpu',
                dict(model_validation=2,aux_gpu=100),omit_receipts=('model_validation',))
            with self.assertRaisesRegex(RuntimeError,'Missing parallel completion receipt'):self.run_pair(owner,cpu,gpu)
            self.assertNotIn('extension_gate',[s[0] for s in owner.starts]);self.assertTrue(owner.children[1].terminated)

    def test_STOP_cleans_both_owned_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp),durations=dict(actual_link=100,P1024=100));owner.stop_at=2
            with self.assertRaisesRegex(RuntimeError,'fixture STOP'):self.run_pair(owner,cpu,gpu)
            self.assertTrue(all(p.terminated for p in owner.children));self.assertFalse(Path(cpu[0]['receipt']).exists())

    def test_parallel_status_keeps_both_children_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp))
            owner.start(cpu[0]['name'],cpu[0]['args'],False);owner.start(gpu['name'],gpu['args'],True)
            snapshot=read(Path(tmp)/'active_child.json');self.assertEqual(snapshot['name'],'parallel')
            self.assertIsNone(snapshot['pid']);self.assertEqual({r['name'] for r in snapshot['children']},{'actual_link','P1024'})
            owner.cleanup_children()

    def test_receiver_timing_cannot_overlap_cpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp),gpu_name='receiver_cost')
            with self.assertRaisesRegex(RuntimeError,'isolated execution'):self.run_pair(owner,cpu,gpu)
            self.assertEqual(owner.starts,[])

    def test_rejects_second_gpu_in_cpu_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp));cpu[0]['gpu']=True
            with self.assertRaisesRegex(RuntimeError,'one CPU chain'):self.run_pair(owner,cpu,gpu)
            self.assertEqual(owner.starts,[])

    def test_delivery_resolution_preserves_all_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);owner,_,_=self.setup_owner(folder)
            item=dict(name='P1024',gpu=True,completion_relative='p/completion.json',
                args=['{CODE}/p.py','{ROOT}','{OUT}','{POLICIES}','{BLER}','{QUALITY_BUNDLE}','{QUALITY_COMPLETION}','{CODEBOOK}'])
            step=owner.delivery_step(item,'selected','bler','bundle',folder/'q','codebook')
            self.assertEqual(step['args'][3:],['selected','bler','bundle',str(folder/'q/completion.json'),'codebook'])
            self.assertEqual(step['receipt'],folder/'p/completion.json');self.assertTrue(step['gpu'])
            self.assertEqual(step['expected_status'],'P1024_BASELINE_EVALUATION_COMPLETE')

    def test_failed_existing_receipt_rejected_before_either_worker_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp));write(cpu[0]['receipt'],dict(status='FAILED'))
            with self.assertRaisesRegex(RuntimeError,'Invalid parallel completion status'):self.run_pair(owner,cpu,gpu)
            self.assertEqual(owner.starts,[])

    def test_changed_existing_output_rejected_before_gpu_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);owner,cpu,gpu=self.setup_owner(folder);output=folder/'data.txt';output.write_text('original')
            write(gpu['receipt'],dict(status='FIXTURE_COMPLETE',outputs={str(output):sha(output)}));output.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Parallel completion output changed'):self.run_pair(owner,cpu,gpu)
            self.assertEqual(owner.starts,[])

    def test_successful_existing_hashed_output_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);owner,cpu,gpu=self.setup_owner(folder);output=folder/'data.txt';output.write_text('original')
            write(gpu['receipt'],dict(status='FIXTURE_COMPLETE',outputs={str(output):sha(output)}))
            self.run_pair(owner,cpu,gpu);self.assertEqual([s[0] for s in owner.starts],['actual_link'])

    def test_new_wrong_status_stops_parallel_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,cpu,gpu=self.setup_owner(Path(tmp),durations=dict(actual_link=2,P1024=100))
            owner.specifications['actual_link']['receipt_value']=dict(status='FAILED')
            with self.assertRaisesRegex(RuntimeError,'Invalid parallel completion status'):self.run_pair(owner,cpu,gpu)
            self.assertTrue(owner.children[1].terminated)

    def test_new_changed_output_stops_parallel_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);owner,cpu,gpu=self.setup_owner(folder,durations=dict(actual_link=2,P1024=100))
            output=folder/'data.txt';output.write_text('original');digest=sha(output);output.write_text('changed')
            owner.specifications['actual_link']['receipt_value']=dict(status='FIXTURE_COMPLETE',outputs={str(output):digest})
            with self.assertRaisesRegex(RuntimeError,'Parallel completion output changed'):self.run_pair(owner,cpu,gpu)
            self.assertTrue(owner.children[1].terminated)

    def test_coarse_merge_starts_only_when_all_shards_complete_and_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);owner,_,_=self.setup_owner(folder);calls=[]
            def merge(phase,destination):
                calls.append((phase,destination));write(destination,dict(status='COARSE_COMPLETE'))
            with patch.object(owner,'merge',side_effect=merge):
                self.assertFalse(owner.merge_coarse_if_ready())
                for d in owner.shards[:-1]:write(d/'coarse_completion.json',dict(status='COARSE_COMPLETE'))
                self.assertFalse(owner.merge_coarse_if_ready())
                write(owner.shards[-1]/'coarse_completion.json',dict(status='COARSE_COMPLETE'))
                self.assertTrue(owner.merge_coarse_if_ready());self.assertFalse(owner.merge_coarse_if_ready())
            self.assertEqual(calls,[('coarse',folder/'bler_coarse.json')])

    def test_incomplete_cpu_merge_cannot_be_called_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner,_,_=self.setup_owner(Path(tmp))
            for d in owner.shards:write(d/'coarse_completion.json',dict(status='COARSE_COMPLETE'))
            with patch.object(owner,'merge',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'without its immutable table'):owner.merge_coarse_if_ready()


if __name__=='__main__':unittest.main()
