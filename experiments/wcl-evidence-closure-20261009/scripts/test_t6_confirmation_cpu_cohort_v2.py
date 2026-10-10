"""Real subprocess scheduling tests with synthetic children only."""
import json,os,sqlite3,sys,tempfile,time,unittest
from contextlib import closing,redirect_stdout
from unittest import mock
import io
from pathlib import Path
import run_t6_confirmation_cpu_cohort_v2 as c

CHILD='''import argparse,json,os,sqlite3,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--branch');p.add_argument('--index');p.add_argument('--fail',action='store_true');p.add_argument('--sleep',type=float,default=.1);a=p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
if a.fail:raise SystemExit(7)
d=sqlite3.connect(str(Path(a.root)/'synthetic_ledger.sqlite'),timeout=30)
d.execute('BEGIN IMMEDIATE');d.execute('INSERT INTO calls VALUES (?,?,?)',(a.branch,int(a.index),os.getpid()));d.commit();d.close()
time.sleep(a.sleep)
'''

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.child=self.root/'synthetic_child.py';self.child.write_text(CHILD)
        with closing(sqlite3.connect(self.root/'synthetic_ledger.sqlite'))as db:
            db.execute('CREATE TABLE calls(branch TEXT,i INTEGER,pid INTEGER,PRIMARY KEY(branch,i))');db.commit()
    def reg(self):
        out=self.root/'cohort';out.mkdir()
        return dict(record_dir=str(out),root=str(self.root),environment=dict(CUDA_VISIBLE_DEVICES=''),deadline_unix=time.time()+30,
            stop_files=[],launch_stagger_seconds=.01,poll_seconds=.005,terminate_timeout_seconds=2)
    def jobs(self):
        return[dict(job_id=f'{b}_{i}',branch=b,index=i,argv=[sys.executable,str(self.child),'--root',str(self.root),'--branch',b,'--index',str(i)])
            for i in range(4)for b in('raw','entropy')]
    def request(self):
        ids=[f'synthetic{i}'for i in range(100)];out=self.root/'outputs';out.mkdir()
        runner=Path(c.__file__).with_name('t6_confirmation_render.py').resolve()
        r=dict(schema=c.REQUEST_SCHEMA,N=2048,source_count=100,frame_count=2700,packet_cap=5400,workers=4,
            snrs=[4,10,19],noise_seeds=[9201,9202,9203],source_ids=ids,records=[dict(source_index=i,source_id=x)for i,x in enumerate(ids)],
            source_reselection=False,policy_reselection=False,schedule=[dict(branch='raw')for _ in range(6)]+[dict(branch='entropy')for _ in range(3)],
            out=str(out),root=str(self.root),python_cpu=sys.executable,source_bindings={str(runner):c.sha(runner)},deadline_unix=time.time()+30,stop_files=[])
        p=out/'request.json';c.save(p,r);return p,r
    def test_eight_real_children_separate_branch_processes_share_one_ledger(self):
        reg=self.reg();result=c.children_stage(reg,self.jobs(),'workers')
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(result['worker_exit_codes'],[0]*8)
        with closing(sqlite3.connect(self.root/'synthetic_ledger.sqlite'))as db:rows=db.execute('SELECT branch,i,pid FROM calls').fetchall()
        self.assertEqual(len(rows),8);self.assertEqual(len({x[2]for x in rows}),8)
        self.assertEqual({(b,i)for b,i,_ in rows},{(b,i)for b in('raw','entropy')for i in range(4)})
        for job in self.jobs():
            launch=c.read(Path(reg['record_dir'])/job['job_id']/'launch.json');exit=c.read(Path(reg['record_dir'])/job['job_id']/'exit.json')
            self.assertEqual(launch['argv'],job['argv']);self.assertEqual(launch['child_pid'],exit['child_pid']);self.assertTrue(exit['actual_child_waited'])
    def test_failed_first_child_stops_remaining_launches_and_no_retry(self):
        reg=self.reg();reg['launch_stagger_seconds']=.3;jobs=self.jobs();jobs[0]['argv'].append('--fail')
        result=c.children_stage(reg,jobs,'workers');self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['launched_children'],1);self.assertEqual(result['worker_exit_codes'],[7]);self.assertEqual(len(result['unlaunched_jobs']),7)
        with self.assertRaises(FileExistsError):c.children_stage(reg,jobs,'workers')
    def test_prelaunch_stop_has_zero_children(self):
        reg=self.reg();stop=self.root/'STOP';stop.touch();reg['stop_files']=[str(stop)]
        result=c.children_stage(reg,self.jobs(),'workers');self.assertEqual(result['launched_children'],0);self.assertEqual(result['status'],'FAILED')
    def test_failed_sibling_is_stopped_and_all_actual_exits_waited(self):
        reg=self.reg();reg['launch_stagger_seconds']=.15;jobs=self.jobs();jobs[0]['argv']+=['--sleep','10'];jobs[1]['argv'].append('--fail')
        result=c.children_stage(reg,jobs,'workers');self.assertEqual(result['status'],'FAILED');self.assertEqual(result['launched_children'],2)
        self.assertEqual(len(result['exits']),2);self.assertNotEqual(result['worker_exit_codes'][0],0);self.assertEqual(result['worker_exit_codes'][1],7)
        self.assertTrue(all(x['actual_child_waited']for x in result['exits']));self.assertEqual(len(result['unlaunched_jobs']),6)
    def test_registration_seals_every_dependency_and_fixed_four_per_arm(self):
        request,r=self.request();out=Path(r['out'])/'cohort';result=c.prepare(request,out);reg=c.read(result['registration']['path'])
        self.assertEqual(reg['total_workers'],8);self.assertEqual(reg['packet_cap'],5400);self.assertEqual(reg['source_bindings'],r['source_bindings'])
        self.assertTrue(all('--branch'in x['argv']for x in reg['jobs']));self.assertEqual(reg['environment']['CUDA_VISIBLE_DEVICES'],'')
        with self.assertRaisesRegex(RuntimeError,'Fresh cohort'):c.prepare(request,out)
        r['workers']=8
        with self.assertRaisesRegex(RuntimeError,'workers 4'):c.validate(r)
    def test_different_record_dir_cannot_restart_existing_global_attempt(self):
        request,r=self.request();result=c.prepare(request,Path(r['out'])/'cohort');claim=Path(r['out'])/'cpu_cohort_launch_claim.json';c.save(claim,dict(existing=True))
        with self.assertRaises(FileExistsError):c.run(result['registration']['path'])
        with self.assertRaisesRegex(RuntimeError,'already attempted'):c.prepare(request,Path(r['out'])/'another_cohort')
        self.assertFalse((Path(r['out'])/'cohort'/'ledger_init').exists())
    def test_mutated_request_rejected_before_global_claim(self):
        request,r=self.request();result=c.prepare(request,Path(r['out'])/'cohort');request.write_text(request.read_text()+' ')
        with self.assertRaisesRegex(RuntimeError,'Changed sealed dependency'):c.run(result['registration']['path'])
        self.assertFalse((Path(r['out'])/'cpu_cohort_launch_claim.json').exists())

    def test_interpreter_symlink_semantics_preserved_in_prepare_and_run(self):
        # A venv interpreter is identified by its lexical path beside pyvenv.cfg,
        # even when resolving the link points to the base interpreter.
        request,r=self.request();alias=self.root/'synthetic_venv'/'bin'/'python'
        alias.parent.mkdir(parents=True);(alias.parent.parent/'pyvenv.cfg').write_text('synthetic semantic fixture only\n')
        base=Path(sys.executable).resolve();r['python_cpu']=str(alias)
        request.write_text(json.dumps(r));original_resolve=Path.resolve;original_is_file=Path.is_file
        def resolved(path,*args,**kwargs):
            return base if os.path.abspath(str(path))==os.path.abspath(str(alias)) else original_resolve(path,*args,**kwargs)
        def is_file(path):
            return True if os.path.abspath(str(path))==os.path.abspath(str(alias)) else original_is_file(path)
        observed=[]
        def synthetic_stage(reg,jobs,phase):
            observed.append((phase,jobs))
            self.assertTrue(all(j['argv'][0]==os.path.abspath(str(alias)) for j in jobs))
            if phase=='ledger_init':c.save(Path(r['out'])/'ledger_initialized.json',dict(request_sha256=c.sha(request),packet_cap=5400))
            return dict(status='COMPLETE',worker_exit_codes=[0]*len(jobs))
        with mock.patch.object(Path,'resolve',resolved),mock.patch.object(Path,'is_file',is_file):
            self.assertEqual(alias.resolve(),base)
            info=c.prepare(request,Path(r['out'])/'cohort');reg=c.read(info['registration']['path'])
            self.assertEqual(reg['python_cpu'],os.path.abspath(str(alias)))
            self.assertNotEqual(reg['python_cpu'],str(base))
            self.assertEqual(reg['ledger_init']['argv'][0],reg['python_cpu'])
            self.assertTrue(all(j['argv'][0]==reg['python_cpu']for j in reg['jobs']))
            with mock.patch.object(c,'children_stage',side_effect=synthetic_stage),redirect_stdout(io.StringIO()):
                self.assertEqual(c.run(info['registration']['path']),0)
        self.assertEqual([(phase,len(jobs))for phase,jobs in observed],[('ledger_init',1),('workers',8)])
        self.assertEqual(c.read(Path(r['out'])/'cohort'/'exit.json')['shared_packet_cap'],5400)

    def test_actual_symlink_interpreter_registration_keeps_alias(self):
        request,r=self.request();alias=self.root/'actual_symlink_venv'/'bin'/'python';alias.parent.mkdir(parents=True)
        try:alias.symlink_to(Path(sys.executable).resolve())
        except OSError as error:self.skipTest('Host does not permit unprivileged file symlinks: '+str(error))
        self.assertNotEqual(str(alias),str(alias.resolve()))
        r['python_cpu']=str(alias);request.write_text(json.dumps(r))
        info=c.prepare(request,Path(r['out'])/'cohort');reg=c.read(info['registration']['path'])
        self.assertEqual(reg['python_cpu'],os.path.abspath(str(alias)))
        self.assertTrue(all(j['argv'][0]==os.path.abspath(str(alias))for j in reg['jobs']))

if __name__=='__main__':unittest.main()
