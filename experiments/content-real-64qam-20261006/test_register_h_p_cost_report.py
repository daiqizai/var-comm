import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h_p_cost_report_driver as d
import register_h_p_cost_report as r


def context(root):
    base=root/'H';base.mkdir();old=dict(root=str(root),out=str(base),owner_out=str(base/'oldexec'),registration='old',
        phase_limits={'development':13200},cpu_affinities=[[0,1],[2,3]],gpu_affinity=[4,5,6,7,8,9],gpu_threads=6,stages=[])
    return dict(template_owner=old,python='/fixture/UNIFIED-METRICS-test/bin/python',
        batches={'H':dict(data_dir=base/'oldresults',closed={'owner':old})})


class Registrar(unittest.TestCase):
    def test_exact_one_CPU_stage_and_original_resources_retained(self):
        with tempfile.TemporaryDirectory() as td:
            x=context(Path(td));old=copy.deepcopy(x['template_owner']);base=Path(old['out'])
            req=dict(execution_dir=str(base/'newexec'),out=str(base/'newout'),qualification='/q/completion.json',max_seconds=1800)
            cfg,own=r.make_configs(req,x);s=own['stages'][0];job=s['jobs'][0]
            self.assertEqual((s['id'],s['resource'],len(s['jobs'])),('report','cpu',1))
            self.assertEqual(job['receipt_expect'],d.EXPECTED);self.assertEqual(job['environment']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(own['cpu_affinities'],old['cpu_affinities']);self.assertEqual(own['phase_limits'],old['phase_limits'])
            self.assertEqual(x['template_owner'],old);self.assertEqual(cfg['qualification_test_count'],24)

    def test_prior_output_and_nested_original_directory_refused(self):
        with tempfile.TemporaryDirectory() as td:
            x=context(Path(td));base=Path(x['template_owner']['out'])
            req=dict(execution_dir=str(base/'e'),out=str(base/'out'))
            r.fresh_outputs(req,x);(base/'out').mkdir()
            with self.assertRaises(RuntimeError):r.fresh_outputs(req,x)
            req['out']=str(base/'oldresults/new')
            with self.assertRaises(RuntimeError):r.fresh_outputs(req,x)

    def test_output_must_be_inside_H_and_disjoint(self):
        with tempfile.TemporaryDirectory() as td:
            x=context(Path(td));base=Path(x['template_owner']['out'])
            for out in (base/'e/out',Path(td)/'outside'):
                with self.assertRaises(RuntimeError):r.fresh_outputs(dict(execution_dir=str(base/'e'),out=str(out)),x)

    def test_no_actual_registration_on_nonlinux(self):
        with patch.object(r.sys,'platform','win32'):
            with self.assertRaisesRegex(RuntimeError,'Linux'):r.register('/missing/request.json')

    def test_finite_deadline_required(self):
        with tempfile.TemporaryDirectory() as td:
            x=context(Path(td));base=Path(x['template_owner']['out'])
            for maximum in (0,7201,float('inf')):
                with self.assertRaises(RuntimeError):r.make_configs(dict(execution_dir=str(base/'e'),out=str(base/'out'),
                    qualification='/q/completion.json',max_seconds=maximum),x)

    def test_qualification_other_request_and_nonzero_exit_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rp=root/'request.json';d.save(rp,{});qp=root/'completion.json'
            q=dict(status=d.QUALIFIED,python='/UM/python',GPU_used=False,new_packet_decodes=0,actual_quality_rows_read=False,
                test_count=24,results=[dict(argv=['/UM/python','-B','-m','unittest','-v',*d.QUALIFICATION_MODULES],exit_code=0,test_count=24)],
                input_bindings={},source_bindings={},outputs={})
            d.save(qp,q);x=dict(python='/UM/python')
            with self.assertRaisesRegex(RuntimeError,'exact registration request'):r.qualify_receipt(rp,{'qualification':str(qp)},x)
            q['results'][0]['exit_code']=1;qp.write_text(__import__('json').dumps(q),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'Exact CPU'):r.qualify_receipt(rp,{'qualification':str(qp)},x)


if __name__=='__main__':unittest.main()
