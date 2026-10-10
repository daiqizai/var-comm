"""Synthetic metadata/byte/owner failures only. No real pool or image access."""
import copy
import hashlib
import io as byteio
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ep_new100_content_io_v1 as support
import ep_new100_selection_owner_v1 as selection
import ep_new100_download_owner_v1 as download
import ep_new100_content_owner_v1 as content
import ep_source_population as population


def fixed100():
    records=[];headers={}
    for i in range(100):
        sid='n00000001/ILSVRC2012_val_%08d_n00000001'%(i+1)
        records.append(dict(source_index=i,source_id=sid,canonical_source_id='imagenet-val:%08d'%(i+1),
            path='/synthetic/'+sid+'.JPEG',original_bytes=3))
        headers[i+1]=dict(name='ILSVRC2012_val_%08d.JPEG'%(i+1),data_offset=512+i*1024,bytes=3,header_sha256='a'*64)
    selected=dict(source_count=100,records=records,source_ids=[r['source_id'] for r in records])
    plan=selection.download_plan(selected,headers,{'n00000001':0},{'path':'s','sha256':'1'*64,'bytes':10},
        {'path':'p','sha256':'2'*64,'bytes':10},{'path':'r','sha256':'3'*64,'bytes':10},{})
    return selected,headers,plan


class Response:
    def __init__(self,status=206,headers=None,data=b'abc'):
        self.status=status;self.headers=headers or {'ETag':selection.ETAG,'Content-Range':f'bytes 512-514/{selection.TOTAL}',
            'Content-Length':'3'};self.data=byteio.BytesIO(data);self.reads=0
    def getheader(self,key):return self.headers.get(key)
    def read(self,n):self.reads+=1;return self.data.read(n)


class Connection:
    def __init__(self,response):self.response=response;self.calls=[];self.closed=False
    def request(self,*args,**kwargs):self.calls.append((args,kwargs))
    def getresponse(self):return self.response
    def close(self):self.closed=True


class DownloadTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_full_archive200_never_read(self):
        response=Response(status=200);connection=Connection(response)
        with self.assertRaisesRegex(ValueError,'Exact206'):
            download.fetch_one({'data_offset':512,'bytes':3},self.root/'one',time.time()+20,lambda *a,**k:connection)
        self.assertEqual(response.reads,0);self.assertFalse((self.root/'one').exists());self.assertTrue(connection.closed)
    def test_wrong_etag_never_read(self):
        response=Response();response.headers['ETag']='different'
        with self.assertRaises(ValueError):download.fetch_one({'data_offset':512,'bytes':3},self.root/'one',time.time()+20,lambda *a,**k:Connection(response))
        self.assertEqual(response.reads,0)
    def test_wrong_range_never_read(self):
        response=Response();response.headers['Content-Range']='bytes 513-515/999'
        with self.assertRaises(ValueError):download.fetch_one({'data_offset':512,'bytes':3},self.root/'one',time.time()+20,lambda *a,**k:Connection(response))
        self.assertEqual(response.reads,0)
    def test_short_payload_preserved_no_retry(self):
        response=Response(data=b'ab');connection=Connection(response)
        with self.assertRaisesRegex(ValueError,'Incomplete payload'):
            download.fetch_one({'data_offset':512,'bytes':3},self.root/'one',time.time()+20,lambda *a,**k:connection)
        self.assertEqual((self.root/'one').read_bytes(),b'ab');self.assertEqual(len(connection.calls),1)
    def test_exact_payload_bytes_and_one_request(self):
        connection=Connection(Response());result=download.fetch_one({'data_offset':512,'bytes':3},self.root/'one',time.time()+20,lambda *a,**k:connection)
        self.assertEqual(result,dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest()))
        self.assertEqual(len(connection.calls),1);self.assertEqual(connection.calls[0][1]['headers']['Range'],'bytes=512-514')
    def test_no_overwrite_payload(self):
        (self.root/'one').write_bytes(b'old')
        with self.assertRaises(FileExistsError):download.fetch_one({'data_offset':512,'bytes':3},self.root/'one',time.time()+20,lambda *a,**k:Connection(Response()))
        self.assertEqual((self.root/'one').read_bytes(),b'old')
    def test_modified_byte_budget_rejected(self):
        _,_,plan=fixed100();plan['maximum_payload_bytes']+=1
        with self.assertRaisesRegex(ValueError,'bounded bytes'):download.validate_plan(plan)
    def test_duplicate_selected_id_rejected(self):
        _,_,plan=fixed100();plan['records'][1]['canonical_source_id']=plan['records'][0]['canonical_source_id']
        with self.assertRaises(ValueError):download.validate_plan(plan)
    def test_fullsize_length_mismatch_stops_without_alternative(self):
        selected,headers,_=fixed100();headers[4]['bytes']=7
        with self.assertRaisesRegex(ValueError,'LENGTH_MISMATCH_STOP_NO_REPLACEMENT'):
            selection.download_plan(selected,headers,{'n00000001':0},{},{},{},{})
        self.assertEqual(len(selected['records']),100)
    def test_original_source_order_mandatory(self):
        selected,headers,_=fixed100();selected['records'][3]['source_index']=4
        with self.assertRaises(ValueError):selection.download_plan(selected,headers,{'n00000001':0},{},{},{},{})


class ContentTarTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def archive(self,change=None,extra=False):
        _,_,plan=fixed100();records=[];path=self.root/'test.tar'
        with tarfile.open(path,'x',format=tarfile.USTAR_FORMAT) as tar:
            for i,row in enumerate(plan['records']):
                raw=bytes([i,0,255]);name='%04d.JPEG'%i;info=tarfile.TarInfo(name);info.size=len(raw)
                if change is not None:change(i,info)
                tar.addfile(info,byteio.BytesIO(raw) if info.isfile() else None)
                records.append(dict(row,transport_member=name,newly_observed_official_JPEG_sha256=hashlib.sha256(raw).hexdigest()))
            if extra:
                info=tarfile.TarInfo('extra.JPEG');info.size=3;tar.addfile(info,byteio.BytesIO(b'bad'))
        return path,dict(archive=support.pin(path),records=records)
    def extract(self,path,transport):
        out=self.root/'images';out.mkdir();return content.verify_tar(path,transport,out,lambda:None)
    def test_exact100_flat_members(self):
        path,transport=self.archive();result=self.extract(path,transport)
        self.assertEqual(len(result),100);self.assertEqual(result[3]['sha256'],transport['records'][3]['newly_observed_official_JPEG_sha256'])
    def test_extra_member_stops(self):
        path,transport=self.archive(extra=True)
        with self.assertRaisesRegex(ValueError,'extra tar member'):self.extract(path,transport)
    def test_traversal_rejected_without_writing(self):
        def mutate(i,info):
            if i==0:info.name='../outside.JPEG'
        path,transport=self.archive(mutate)
        with self.assertRaises(ValueError):self.extract(path,transport)
        self.assertFalse((self.root/'outside.JPEG').exists());self.assertEqual(list((self.root/'images').iterdir()),[])
    def test_link_rejected(self):
        def mutate(i,info):
            if i==0:info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd';info.size=0
        path,transport=self.archive(mutate)
        with self.assertRaises(ValueError):self.extract(path,transport)
    def test_payload_sha_rejected_before_decode(self):
        path,transport=self.archive();transport['records'][0]['newly_observed_official_JPEG_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'payload SHA'):self.extract(path,transport)
    def test_transport_sha_rejected(self):
        path,transport=self.archive();transport['archive']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Transport SHA'):self.extract(path,transport)


class GateAndBoundTests(unittest.TestCase):
    def test_full_gate_uses_complete_reader_and_original_registration(self):
        import types
        runner=object();request_pin={'path':'request','sha256':'a'*64}
        provider=types.SimpleNamespace(engine=lambda:runner,registration=mock.Mock(return_value='registered'),
            evaluation=types.SimpleNamespace(winners=object()))
        r={'full_calibration':{'request':request_pin},'original_policies':{}}
        def check(pins,read,validate,winners):
            self.assertEqual(validate(request_pin),'registered');self.assertEqual(read({'path':'rows'}),'complete rows')
            self.assertIs(winners,provider.evaluation.winners);return 'closure','winners'
        with mock.patch.object(support,'sha',return_value=selection.FULL_OWNER_SHA),\
             mock.patch.object(selection.importlib,'import_module',return_value=provider),\
             mock.patch.object(support,'read',return_value='complete rows') as reader,\
             mock.patch.object(selection.core,'verify_full_calibration',side_effect=check),\
             mock.patch.object(selection.core,'policies',return_value='policies'):
            self.assertEqual(selection.full_gate(r),'policies')
            reader.assert_called_once_with({'path':'rows'},maximum=512<<20)
            provider.registration.assert_called_once_with('request','a'*64,runner)
    def test_read_maximum_transparently_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'json';p.write_text('{"x":1}');desc=support.pin(p)
            self.assertEqual(support.read(desc,base=Path(tmp),maximum=512<<20),{'x':1})
            with self.assertRaises(ValueError):support.read(desc,base=Path(tmp),maximum=2)
    def test_failed_full_policy_never_ranks_pool(self):
        r={'out':'/synthetic','cpu_slots':[2,3]}
        with mock.patch.object(selection,'registration',return_value=r),mock.patch.object(support,'guard'),\
             mock.patch.object(selection,'closed_registry',return_value={}),mock.patch.object(selection,'full_gate',side_effect=ValueError('not closed')),\
             mock.patch.object(support,'inside',side_effect=Path),mock.patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),\
             mock.patch.object(selection.os,'sched_getaffinity',return_value={2,3},create=True),mock.patch.object(population,'select_population') as rank:
            with self.assertRaisesRegex(ValueError,'not closed'):selection.worker({},123)
            rank.assert_not_called()
    def test_failed_registry_never_reads_full_policy_or_ranks(self):
        r={'out':'/synthetic','cpu_slots':[2,3]}
        with mock.patch.object(selection,'registration',return_value=r),mock.patch.object(support,'guard'),\
             mock.patch.object(selection,'closed_registry',side_effect=ValueError('not closed')),mock.patch.object(selection,'full_gate') as policy,\
             mock.patch.object(support,'inside',side_effect=Path),mock.patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),\
             mock.patch.object(selection.os,'sched_getaffinity',return_value={2,3},create=True),mock.patch.object(population,'select_population') as rank:
            with self.assertRaises(ValueError):selection.worker({},123)
            rank.assert_not_called();policy.assert_not_called()
    def test_fixed_two_CPU_and_no_retry(self):
        r=dict(max_seconds=600,deadline_unix=time.time()+300,cpu_slots=[2,3],automatic_retry=False,automatic_successor=False,out='/synthetic')
        with mock.patch.object(support,'inside'):
            self.assertEqual(support.validate_execution(r),[2,3]);r['cpu_slots']=[2,3,4]
            with self.assertRaises(ValueError):support.validate_execution(r)
            r['cpu_slots']=[2,3];r['automatic_retry']=True
            with self.assertRaises(ValueError):support.validate_execution(r)
    def test_no_unbounded_or_expired_window(self):
        r=dict(max_seconds=3601,deadline_unix=time.time()+300,cpu_slots=[2,3],automatic_retry=False,automatic_successor=False,out='/synthetic')
        with self.assertRaises(ValueError):support.validate_execution(r)
        r['max_seconds']=600;r['deadline_unix']=time.time()-1
        with self.assertRaises(ValueError):support.validate_execution(r)
    def test_changed_pin_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{}');desc=support.pin(p);p.write_text('[]')
            with self.assertRaisesRegex(ValueError,'Pinned bytes changed'):support.read(desc,base=Path(tmp))
    def test_pin_reads_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a';p.write_bytes(b'1234')
            with self.assertRaisesRegex(ValueError,'Bounded'):support.bytes_checked(support.pin(p),maximum=3,base=Path(tmp))
    def test_duplicate_canonical_pixels_blocks_gate(self):
        selected,_,_=fixed100();rp={'path':'r','sha256':'1'*64,'bytes':1};sp={'path':'s','sha256':'2'*64,'bytes':1}
        selected['registry']=rp
        reg=dict(full_canonical_content_dedup_ready=True,study_scope_closed=True,pixel_hash_missing_count=0,
            records=[dict(original_file_sha256=[],pixel_sha256=['a'*64])],canonical_source_ids=['old'],source_count=1)
        receipt=dict(schema='EP_CANDIDATE100_CONTENT_RECEIPT_V1',selection=sp,pixel_hash_domain=population.PIXEL_DOMAIN,
            Encoder_calls_before_check=0,VAR_calls_before_check=0,metric_calls_before_check=0,records=[])
        for i,row in enumerate(selected['records']):
            receipt['records'].append(dict(source_id=row['source_id'],source_index=i,original_file_sha256=hashlib.sha256(('raw'+str(i)).encode()).hexdigest(),
                pixel_sha256='a'*64 if i==7 else hashlib.sha256(('pre'+str(i)).encode()).hexdigest(),
                horizontal_flip_pixel_sha256=hashlib.sha256(('flip'+str(i)).encode()).hexdigest()))
        gate=population.check_content(reg,rp,selected,sp,receipt,{'path':'c','sha256':'3'*64,'bytes':1})
        self.assertEqual(gate['status'],'EP_NEW100_CONTENT_DUPLICATE_STOP');self.assertFalse(gate['Encoder_calls_allowed'])
        self.assertFalse(gate['source_reselection_allowed']);self.assertEqual(gate['conflicts'][0]['source_index'],7)


class ActualOwnerControlTests(unittest.TestCase):
    def exercise(self,rc):
        import types
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);registered=root/'registered';registered.mkdir()
        r=dict(schema='SYNTHETIC',out=str(registered),cpu_slots=[2,3],claim_name='once.json',max_seconds=10,
            deadline_unix=time.time()+20,automatic_retry=False,automatic_successor=False,worker_python='unused')
        request={'path':str(registered/'request.json'),'sha256':'a'*64};support.save(request['path'],r)
        done=dict(status='SYNTHETIC_PASS',request_sha256='a'*64,results={'synthetic':True})
        fake=types.SimpleNamespace(pid=99999,returncode=rc,wait=lambda **k:rc,poll=lambda:rc)
        def popen(*a,**k):support.save(registered/'run/worker_completion.json',done);return fake
        original_read=support.read
        def read(desc,*a,**k):return original_read(desc,base=root)
        patches=[mock.patch.object(support.sys,'platform','linux'),mock.patch.object(support,'RT',root),
            mock.patch.object(support,'inside',side_effect=lambda p,*a,**k:Path(p)),mock.patch.object(support.os,'sched_getaffinity',return_value={2,3},create=True),
            mock.patch.object(support,'controlled_environment',return_value={}),mock.patch.object(support.subprocess,'Popen',side_effect=popen),
            mock.patch.object(support,'read',side_effect=read)]
        import contextlib
        with contextlib.ExitStack() as stack:
            for patch in patches:stack.enter_context(patch)
            if rc:
                with self.assertRaisesRegex(ValueError,'Actual child failed'):support.run_owner(request,Path('unused'),lambda _:r,'SYNTHETIC_PASS')
            else:
                result=support.run_owner(request,Path('unused'),lambda _:r,'SYNTHETIC_PASS')
                self.assertEqual(result['worker_exit_codes'],[0]);self.assertTrue(result['actual_children_waited'])
            with self.assertRaisesRegex(ValueError,'already attempted'):support.run_owner(request,Path('unused'),lambda _:r,'SYNTHETIC_PASS')
        wait=json.loads((registered/'run/actual_child_wait.json').read_text());self.assertEqual(wait['returncode'],rc)
        self.assertTrue(wait['actual_wait']);return registered
    def test_success_actual_wait_and_no_replay(self):self.exercise(0)
    def test_failure_actual_wait_preserved_and_no_replay(self):
        path=self.exercise(2);failure=json.loads((path/'run/failure.json').read_text())
        self.assertTrue(failure['child_waited']);self.assertEqual(failure['returncode'],2);self.assertTrue(failure['no_retry'])


if __name__=='__main__':unittest.main(verbosity=2)
