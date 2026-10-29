"""Verify frozen V5 evidence and record the V6 editorial/asset-availability revision."""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import re

P = Path(__file__).resolve().parents[1]
OLD = P.parent / 'nestcom_wcl_focused_20261029_v5'
assert OLD.is_dir(), 'Place the delivered V5 directory adjacent to V6.'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(path):
    return {'path': path.relative_to(P).as_posix(), 'sha256': sha(path),
            'bytes': path.stat().st_size}

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

frozen = {}
for folder in ('inputs', 'figures', 'tables'):
    frozen[folder] = []
    for old in sorted((OLD / folder).rglob('*')):
        if not old.is_file():
            continue
        path = P / old.relative_to(OLD)
        assert path.is_file() and sha(path) == sha(old), str(path)
        item = record(path)
        item['parent_path'] = old.relative_to(P.parents[2]).as_posix()
        frozen[folder].append(item)
assert [len(frozen[k]) for k in ('inputs', 'figures', 'tables')] == [56, 35, 7]
assert sha(P / 'sections/method.tex') == sha(OLD / 'sections/method.tex')
assert sha(P / 'supplementary.pdf') == sha(OLD / 'supplementary.pdf')
equations = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',
                       (P / 'sections/method.tex').read_text(), re.S)
assert len(equations) == 6
new_inputs = [record(path) for path in sorted((P / 'inputs').rglob('*'))
              if path.is_file() and not (OLD / path.relative_to(P)).exists()]
assert len(new_inputs) == 2
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
                     (P / 'sections/abstract.tex').read_text(), re.S).group(1)
abstract_words = len(abstract.replace('~', ' ').replace(r'\%', '%').split())
assert 75 <= abstract_words <= 100

science = {'new_training': 0, 'new_neural_inference': 0, 'new_channel_events': 0,
           'new_metric_forwards': 0, 'new_bootstrap_resamples': 0}
manifest = {'revision': 'V6',
    'scientific_base_commit': 'b55bc901a285d91499550290a9146ec1eaeb317f',
    'parent_manuscript': 'paper/manuscripts/' + OLD.name,
    'parent_package_sha256': '9bc12c172e0109e9ea3b91ae18384be6ccbdea7cd60c11ff73c35bbb94613fb6',
    'frozen_parent_files': frozen, 'new_descriptive_inputs': new_inputs,
    'abstract_word_count': abstract_words, 'scientific_execution': science,
    'figure3': {'delivered_snr_db': 10, 'N': 1024,
                'proposed_snr_db': 4, 'proposed_image_status': 'NOT_MATERIALIZED',
                'task': 'Figure3_4dB_Minimal_Task_zh.md'}}
write(P / 'input_manifest.json', manifest)

mapping = json.loads((OLD / 'figure_table_sources.json').read_text())
def relink(value):
    if isinstance(value, str):
        return value.replace(OLD.name, P.name)
    if isinstance(value, list):
        return [relink(v) for v in value]
    if isinstance(value, dict):
        return {k: relink(v) for k, v in value.items()}
    return value
mapping = relink(mapping)
mapping['revision'] = 'V6'
mapping['previous_source_map'] = 'paper/manuscripts/' + OLD.name + '/figure_table_sources.json'
for figure in mapping.get('figures', {}).values():
    figure['unchanged_from'] = 'paper/manuscripts/' + OLD.name
mapping['figure3_4db_status'] = manifest['figure3']
write(P / 'figure_table_sources.json', mapping)

values = list(csv.DictReader((P / 'figures/figure2_data.csv').open()))
lookup = {(r['method'], r['metric'], int(r['snr_db'])): float(r['mean']) for r in values}
assert len(values) == 36
assert round(lookup['NEST', 'lpips_alex', 4], 3) == 0.263
assert round(lookup['NEST', 'dreamsim', 4], 3) == 0.184
assert round(lookup['ADA_PREDCLASS_FIXEDFRAME', 'lpips_alex', 4], 3) == 0.395
assert round(lookup['ADA_PREDCLASS_FIXEDFRAME', 'dreamsim', 4], 3) == 0.243

write(P / 'validation/data_identity_check.json', {
    'status': 'PASS', 'frozen_inputs': 56, 'unchanged_figure_files': 35,
    'unchanged_table_files': 7, 'method_section_byte_identical': True,
    'unchanged_display_equations': 6, 'figure2_points': 36,
    'added_4db_prose_values_verified': True, 'abstract_word_count': abstract_words,
    'supplement_byte_identical': True, 'new_4db_images_generated': False,
    'scientific_execution': science})
write(P / 'validation/publication_status.json', {
    'status': 'USER_WILL_UPLOAD', 'committed': False, 'pushed': False,
    'remote_modified': False,
    'upload_path': 'paper/manuscripts/' + P.name})

names = sorted({x.relative_to(OLD) for x in OLD.rglob('*')
                if x.is_file() and x.suffix in {'.tex', '.py'}} |
               {x.relative_to(P) for x in P.rglob('*')
                if x.is_file() and x.suffix in {'.tex', '.py'}})
chunks = []
for relative in names:
    before = (OLD / relative).read_text().splitlines(True) if (OLD / relative).exists() else []
    after = (P / relative).read_text().splitlines(True) if (P / relative).exists() else []
    if before != after:
        chunks.extend(difflib.unified_diff(before, after,
            fromfile='a/' + OLD.name + '/' + relative.as_posix(),
            tofile='b/' + P.name + '/' + relative.as_posix()))
(P / 'V5_to_V6_changes.diff').write_text(''.join(chunks))
print(json.dumps({'status': 'PASS', 'abstract_words': abstract_words,
                  'inputs': 56, 'figures': 35, 'tables': 7,
                  'new_audit_inputs': len(new_inputs)}))
