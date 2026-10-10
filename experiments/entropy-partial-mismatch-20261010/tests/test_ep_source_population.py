"""Synthetic metadata tests; no image, model, PHY, network, or real source draw."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/ep_source_population.py'
spec = importlib.util.spec_from_file_location('ep_source_population', MODULE)
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


def h(s):
    return ep.digest(s.encode())


def sid(i):
    return f'n00000001/ILSVRC2012_val_{i:08d}_n00000001'


def pin(name):
    return dict(path=name, sha256=h(name))


def pool():
    return dict(records=[dict(source_id=sid(i), path='/images/' + sid(i) + '.JPEG', original_bytes=1)
                         for i in range(1, 105)])


def registry(complete=True):
    return dict(schema='EP_STUDY_SOURCE_EXCLUSION_REGISTRY_V1', study_scope_closed=True,
        full_canonical_content_dedup_ready=complete, pixel_hash_missing_count=0 if complete else 1,
        source_count=1, canonical_source_ids=[ep.canonical_id(sid(1))],
        records=[dict(canonical_source_id=ep.canonical_id(sid(1)),
                      original_file_sha256=[h('old raw')], pixel_sha256=[h('old pixels')])])


def selected_and_receipt():
    r = registry()
    selected = ep.select_population(r, pin('registry'), pool(), pin('pool'))
    receipt = dict(schema='EP_CANDIDATE100_CONTENT_RECEIPT_V1', selection=pin('selection'),
        pixel_hash_domain=ep.PIXEL_DOMAIN, Encoder_calls_before_check=0, VAR_calls_before_check=0,
        metric_calls_before_check=0, records=[dict(source_id=x['source_id'], source_index=x['source_index'],
            original_file_sha256=h(x['source_id']+'raw'), pixel_sha256=h(x['source_id']+'pixels'),
            horizontal_flip_pixel_sha256=h(x['source_id']+'flip')) for x in selected['records']])
    return r, selected, receipt


class MetadataTests(unittest.TestCase):
    def test_alias_canonicalization_and_invalid_canonical(self):
        self.assertEqual(ep.canonical_id('/old/' + sid(1) + '.JPEG'), 'imagenet-val:00000001')
        self.assertEqual(ep.canonical_id('train-a.parquet:000001'), 'train-a.parquet:000001')
        with self.assertRaises(ValueError):
            ep.canonical_id('imagenet-val:1')

    def test_selection_deterministic_excludes_seen_and_ignores_old_pool_flags(self):
        p = pool()
        for row in p['records']:
            row.update(excluded=False, selection_key='irrelevant-old-seed')
        a = ep.select_population(registry(), pin('r'), p, pin('p'))
        p['records'].reverse()
        b = ep.select_population(registry(), pin('r'), p, pin('p'))
        self.assertEqual(a['records'], b['records'])
        self.assertNotIn(sid(1), a['source_ids'])
        self.assertEqual((a['source_count'], a['frame_count'], a['max_packet_calls']), (100, 3600, 7200))
        self.assertFalse(a['Encoder_calls_allowed'])
        self.assertEqual(a['noise_seeds'], [9301, 9302, 9303])

    def test_unclosed_scope_and_alias_pool_stop(self):
        r = registry()
        r['study_scope_closed'] = False
        with self.assertRaises(ValueError):
            ep.select_population(r, pin('r'), pool(), pin('p'))
        p = pool()
        p['records'].append(dict(p['records'][0]))
        with self.assertRaises(ValueError):
            ep.select_population(registry(), pin('r'), p, pin('p'))

    def test_complete_hash_gate_and_no_prior_model_calls(self):
        r, selected, receipt = selected_and_receipt()
        result = ep.check_content(r, pin('registry'), selected, pin('selection'), receipt, pin('content'))
        self.assertTrue(result['Encoder_calls_allowed'])
        r['full_canonical_content_dedup_ready'] = False
        r['pixel_hash_missing_count'] = 1
        with self.assertRaises(ValueError):
            ep.check_content(r, pin('registry'), selected, pin('selection'), receipt, pin('content'))
        r = registry()
        receipt['VAR_calls_before_check'] = 1
        with self.assertRaises(ValueError):
            ep.check_content(r, pin('registry'), selected, pin('selection'), receipt, pin('content'))

    def test_old_and_within_population_flip_duplicates_stop_without_reselection(self):
        for kind in ('old_raw', 'old_flip', 'within_raw', 'within_flip'):
            r, selected, receipt = selected_and_receipt()
            if kind == 'old_raw':
                receipt['records'][0]['original_file_sha256'] = h('old raw')
            elif kind == 'old_flip':
                receipt['records'][0]['horizontal_flip_pixel_sha256'] = h('old pixels')
            elif kind == 'within_raw':
                receipt['records'][1]['original_file_sha256'] = receipt['records'][0]['original_file_sha256']
            else:
                receipt['records'][1]['pixel_sha256'] = receipt['records'][0]['horizontal_flip_pixel_sha256']
            result = ep.check_content(r, pin('registry'), selected, pin('selection'), receipt, pin('content'))
            self.assertEqual(result['status'], 'EP_NEW100_CONTENT_DUPLICATE_STOP')
            self.assertFalse(result['Encoder_calls_allowed'])
            self.assertFalse(result['source_reselection_allowed'])

    def test_wrong_order_and_hash_domain_rejected(self):
        r, selected, receipt = selected_and_receipt()
        receipt['records'].reverse()
        with self.assertRaises(ValueError):
            ep.check_content(r, pin('registry'), selected, pin('selection'), receipt, pin('content'))
        receipt['records'].reverse()
        receipt['pixel_hash_domain'] = 'unknown_float_domain'
        with self.assertRaises(ValueError):
            ep.check_content(r, pin('registry'), selected, pin('selection'), receipt, pin('content'))

    def test_build_merge_scope_and_hash_gap_not_silently_passed(self):
        documents = {
            'a.json': dict(status='T6_AUDITED_OLD_CONTENT_HASH_INVENTORY_COMPLETE', records=[
                dict(source_id=sid(1), raw_JPEG_sha256=[h('raw1')], preprocessing_sha256=[], roles=['history'])]),
            'b.json': dict(role='independent_holdout_after_method_freeze', source_images=1, images=[
                dict(image_id=sid(2), file_sha256=h('raw2'), preprocessed_rgb_sha256=h('pixels2'),
                     path='/images/'+sid(2)+'.JPEG')]),
        }
        request = dict(schema='EP_SOURCE_REGISTRY_INPUT_SPEC_V1', inputs=[
            dict(path='a.json', sha256=h('a.json'), format='t6_reference_inventory', role='base'),
            dict(path='b.json', sha256=h('b.json'), format='legacy_holdout_manifest', role='old_holdout')])
        def loader(path, expected):
            self.assertEqual(expected, h(path))
            return documents[path], pin(path)
        r = ep.build_registry(request, pin('spec'), loader)
        self.assertEqual(r['source_count'], 2)
        self.assertEqual(r['pixel_hash_missing_count'], 1)
        self.assertFalse(r['study_scope_closed'])
        documents['scope.json'] = dict(schema='EP_STUDY_SOURCE_SCOPE_CLOSURE_V1',
            status='EP_STUDY_SOURCE_REGISTRIES_CLOSED', unresolved_registry_paths=[], unresolved_source_roles=[],
            source_content_access_cutoff='fixed-test-cutoff', scanned_project_scopes=['synthetic-only'],
            manifest_sha256=[h('a.json'), h('b.json')], unique_source_count=2)
        request['study_scope_closure'] = pin('scope.json')
        r = ep.build_registry(request, pin('spec'), loader)
        self.assertTrue(r['study_scope_closed'])
        self.assertFalse(r['full_canonical_content_dedup_ready'])
        documents['scope.json']['manifest_sha256'].pop()
        with self.assertRaises(ValueError):
            ep.build_registry(request, pin('spec'), loader)

    def test_normalized_hash_domain_and_new_output_only(self):
        value = dict(schema='EP_NORMALIZED_PRIOR_SOURCES_V1', pixel_hash_domain='wrong', source_count=0, records=[])
        with self.assertRaises(ValueError):
            list(ep.imported_rows(value, 'normalized_records'))
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'receipt.json'
            proof = ep.write_new(target, {'status': 'synthetic'})
            self.assertEqual(ep.read_json(target, proof['sha256'])[0], {'status': 'synthetic'})
            with self.assertRaises(ValueError):
                ep.write_new(target, {'status': 'changed'})
            with self.assertRaises(ValueError):
                ep.read_json(target, h('wrong'))

    def test_inspected_rejections_and_prior_catalog_not_candidate_pool(self):
        value = dict(inspected=[dict(image_id=sid(1), file_sha256=h('raw'),
                                    preprocessed_rgb_sha256=h('pixels'), path='/x')],
                     rejections=[dict(image_id=sid(2), reason='unreadable_source', path='/y')])
        rows = list(ep.imported_rows(value, 'legacy_source_integrity'))
        self.assertEqual([r[0] for r in rows], [sid(1), sid(2)])
        self.assertEqual(rows[1][2], [])
        catalog = dict(status='METADATA_ONLY_HOLDOUT_CANDIDATE_CATALOG_NOT_TEST_DATA_ACCESSED',
                       used_val_indices=['00000001'], candidate_paths=[sid(2)])
        self.assertEqual(list(ep.imported_rows(catalog, 'legacy_catalog_inventory'))[0][0],
                         'imagenet-val:00000001')


if __name__ == '__main__':
    unittest.main()
