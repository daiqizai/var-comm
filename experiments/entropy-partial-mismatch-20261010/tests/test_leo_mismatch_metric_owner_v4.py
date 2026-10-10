"""Native identity merge and bounded control tests; no models or real scores."""
from pathlib import Path
import copy
import sys
import types
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import leo_mismatch_metric_owner_v3 as old
import leo_mismatch_metric_owner_v4 as new


class NativeMergeTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(path='/usr/lib/libc.so.6',bytes=42,sha256='a'*64)
        self.original=dict(self.row,observed_lexical_path='/lib/libc.so.6')

    def test_only_lexical_field_difference_admitted_and_preserved(self):
        oldrows=[self.original];newrows=[self.row];before=copy.deepcopy((oldrows,newrows))
        answer=new.merge_native_facts(oldrows,newrows)
        self.assertEqual(answer,oldrows);self.assertEqual((oldrows,newrows),before)
        self.assertIsNot(answer[0],oldrows[0])

    def test_binary_SHA_and_size_conflicts_rejected(self):
        for changed in (dict(self.row,sha256='b'*64),dict(self.row,bytes=43)):
            with self.assertRaisesRegex(RuntimeError,'path/size/SHA'):new.merge_native_facts([self.original],[changed])

    def test_unknown_old_provenance_cannot_be_discarded(self):
        with self.assertRaisesRegex(RuntimeError,'Unknown native provenance'):
            new.merge_native_facts([dict(self.original,classification='other')],[self.row])

    def test_new_PFS_driver_and_noncanonical_paths_rejected(self):
        paths=('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/lib.so','/usr/lib/libcuda.so.1',
            '/usr/lib/libnvidia-ml.so.1','/usr/lib/../libc.so.6','/usr//lib/libc.so.6')
        for path in paths:
            with self.assertRaisesRegex(RuntimeError,'canonical external'):new.merge_native_facts([],[dict(self.row,path=path)])

    def test_duplicates_and_new_unknown_fields_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):new.merge_native_facts([],[self.row,self.row])
        with self.assertRaisesRegex(RuntimeError,'Exact metric'):new.merge_native_facts([],[dict(self.row,observed_lexical_path='/other')])

    def test_new_libraries_append_without_collapsing_old_alias_provenance(self):
        oldrows=[self.original,dict(self.original,observed_lexical_path='/another/libc.so.6')]
        added=dict(self.row,path='/usr/lib/libextra.so.1')
        self.assertEqual(new.merge_native_facts(oldrows,[self.row,added]),oldrows+[added])

    def test_PFS_runtime_rows_unchanged_during_merge(self):
        environment=types.SimpleNamespace(rows={'PFS':dict(bytes=99,sha256='c'*64)},host_native=[self.original])
        before=copy.deepcopy(environment.rows);answer=(None,environment,None,None,{})
        with mock.patch.object(new.g,'validate_spec',return_value=answer),mock.patch.object(new,'native_binding',return_value=({},[self.row])):
            self.assertIs(new.metric_environment({},False),answer)
        self.assertEqual(environment.rows,before);self.assertEqual(environment.host_native,[self.original])

    def test_metric_and_resource_code_objects_and_caps_unchanged(self):
        for name in ('scores','run','worker'):
            self.assertIs(getattr(new,name).__code__,getattr(old,name).__code__)
        self.assertEqual(new.CAPS,old.CAPS);self.assertEqual(new.CAPS['image_scores'],4500)
        self.assertIs(new.metric,old.metric);self.assertNotEqual(new.OUT,old.OUT)
        self.assertIn('v4_attempt1',str(new.OUT));self.assertIn('v3_attempt1',str(new.FAILED_ROOT))
        self.assertIs(new.original.metric_environment,new.metric_environment)
        self.assertIs(new.run.__globals__['metric_environment'],new.metric_environment)
        self.assertIs(new.original.worker.__globals__['metric_environment'],new.metric_environment)

    def test_runtime_identity_explicitly_binds_repair_and_zero_call_failure(self):
        with mock.patch.object(new,'base_runtime_identity',return_value=dict(original='preserved')),\
            mock.patch.object(new,'prepare_failure',return_value=dict(prior_metric_scores=0)):
            identity=new.runtime_identity({},object())
        self.assertEqual(identity['original'],'preserved')
        self.assertEqual(identity['preserved_zero_call_prepare_failure']['prior_metric_scores'],0)
        self.assertEqual(identity['native_fact_merge']['identity_fields'],['path','bytes','sha256'])
        self.assertTrue(identity['native_fact_merge']['original_lexical_provenance_preserved'])
        self.assertEqual(identity['native_fact_merge']['adapter_source']['sha256'],new.g.sha(new.__file__))

    def test_resource_refusal_and_owner_target_are_v4(self):
        self.assertIs(new.engine().device_from_request.__globals__['DeviceAdapter'],new.DeviceAdapter)
        self.assertEqual(new.engine().worker_identity.__globals__['__file__'],new.__file__)
        self.assertNotEqual(new.OUT,new.visual.OUT)


if __name__=='__main__':unittest.main()
