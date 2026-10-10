"""Budgeted first normal-frame H800 relocation check, not an extra probe.

The original independent RX completes before any expected token/CDF is read.
The resulting normal-frame image is cached and reused without a second decode.
Only this witness's numeric compatibility is asserted; no cross-host claim.
"""
import ast
import copy
import inspect
from pathlib import Path
import numpy as np
import ep_new100_visual_core_v1 as original

require=original.require
CAPS=original.CAPS
POPULATION=original.POPULATION
pilot=original.pilot
confirmation=original.confirmation
raw=original.raw
check_counts=original.check_counts
CORE_SHA='d073c139cf23abce4f538b6ff32ca9744df5c10f95edd930180643600e3b6d71'


def select_witness(logical,sources,read,inside):
    """First clean actual entropy payload in fixed order, never image quality."""
    for row in logical:
        if row['provider_group']=='raw':continue
        packet=read(row['physical_frame']);rx=packet['actual_RX'];p=rx['rx_profile'];body=rx['body']
        if not rx['header']['header_ok'] or body is None or not body['crc_accepted'] or not body['parser_accepted']:continue
        if p['family'] not in ('EC_VAR_WHOLE','EC_VAR_PARTIAL'):continue
        item=sources[row['source_index']];path=inside(item['archive']['path'])
        require(original.pilot.sha(path)==item['archive']['sha256'],'Actual source TX archive changed')
        key=f"m{p['m']}_K{p['K']}"
        with np.load(path,allow_pickle=False) as z:
            if key not in z.files:continue
            sent=z[key].copy()
        received=np.asarray(body['payload'],dtype=np.uint8)
        if sent.dtype!=np.uint8 or not np.array_equal(sent,received):continue
        return dict(frame_index=row['frame_index'],physical_frame=row['physical_frame'],source_index=row['source_index'],
            source_id=row['source_id'],source_assets=item['source_assets'],source_metadata=item['metadata'],TX_archive=item['archive'],
            family=p['family'],m=p['m'],K=p['K'],actual_payload_sha256=original.raw.g.image_sha(received),
            selection='first actual accepted entropy payload equal to actual TX bytes in fixed logical order',
            extra_RX_calls=0,extra_render_calls=0,source_truth_supplied_to_RX=False)
    raise RuntimeError('No normal actual accepted TX-identical entropy frame available for bounded relocation witness')


def assert_after_RX(witness,evidence,read,inside,image_sha,write,out):
    """External diagnostic only; no token/CDF can enter the completed provider."""
    # Persist actual RX evidence first. A mismatch is a stopped engineering run,
    # not a falsely reported channel gray/failure or a truth-corrected decoder.
    path=Path(out)/'actual_relocation_witness.json'
    observed=dict(witness=witness,actual_RX_evidence=copy.deepcopy(evidence),image_sha256=image_sha,
        status='ACTUAL_RX_COMPLETED_BEFORE_EXPECTED_TOKEN_READ',extra_RX_calls=0,extra_render_calls=0)
    write(path,observed)
    require(evidence['source_decode_called'] and evidence['kind']=='tokens' and evidence['render_called'] and
        evidence['transmitted_CDF_used'] is False and evidence['comparison_truth_used'] is False,
        'Device relocation normal-frame source decode incompatible; preserve paid work and stop')
    m,K=witness['m'],witness['K'];count=original.raw.g.OFFSETS[m]+K
    source_path=inside(witness['source_assets']['path'])
    require(original.pilot.sha(source_path)==witness['source_assets']['sha256'],'Post-RX source witness changed')
    with np.load(source_path,allow_pickle=False) as z:expected=z['tokens'][:count].copy()
    actual=np.asarray(evidence['received_tokens'],dtype=np.int64)
    require(expected.dtype==np.int64 and expected.shape==(count,) and actual.shape==(count,), 'Post-RX witness token shape differs')
    meta=read(witness['source_metadata']);scales=m+(K>0)
    tx=meta['CDF_trace'][:scales];rx=evidence['actual_RX_CDF_trace']
    tokens_equal=np.array_equal(expected,actual)
    cdfs_equal=len(tx)==len(rx)==scales and [x['scale'] for x in tx]==[x['scale'] for x in rx]==list(range(scales)) and \
        all(a['cdf_sha256']==b['cdf_sha256'] for a,b in zip(tx,rx))
    result=dict(status='NORMAL_FRAME_RELOCATION_WITNESS_PASS' if tokens_equal and cdfs_equal else 'NORMAL_FRAME_RELOCATION_WITNESS_MISMATCH',
        actual_RX=original.pilot.descriptor(path),witness=witness,tokens_exact=tokens_equal,CDFs_exact=cdfs_equal,
        checked_source_tokens=int(count),checked_CDF_scales=scales,expected_token_sha256=original.raw.g.image_sha(expected),
        received_token_sha256=original.raw.g.image_sha(actual),extra_RX_calls=0,extra_render_calls=0,
        normal_frame_result_reused=True,expected_values_read_after_RX=True,truth_correction=False,
        scope='This actual normal-frame witness only; no general cross-device bitwise claim')
    proof=Path(out)/'relocation_witness_check.json';write(proof,result)
    require(tokens_equal and cdfs_equal,'Device relocation token/CDF mismatch; no further scientific calls admitted')
    return original.pilot.descriptor(proof)


def prime(r,backend,codec,partial,ledger,boundary,out,g,read):
    witness=r['relocation_witness'];row=r['logical_rows'][witness['frame_index']]
    packet=read(row['physical_frame']);actual=original.validate_packet(row,packet,r['providers'])
    key,prepared,status=original.input_and_status(actual,row['provider_group'],r['visual_identity'],None,None)
    boundary();backend.source_index=row['source_index']
    image,evidence=original.reconstruct(actual,row['provider_group'],prepared,backend,codec,partial,ledger,boundary)
    proof=assert_after_RX(witness,evidence,read,g.inside,g.image_sha(image),g.write,out)
    path=Path(out)/'images/00000.npz'
    with path.open('xb') as f:np.savez(f,image=image)
    return {key:dict(image_archive=original.pilot.descriptor(path),image_sha256=g.image_sha(image),
        actual_source_input_key=key,evidence=evidence,first_physical_frame=row['physical_frame'],
        origin='THIS_NEW100_FIRST_NORMAL_RELOCATION_FRAME',relocation_witness_check=proof)},key


def images(r,backend,codec,partial,ledger,boundary,out,g,read):
    require(original.pilot.sha(original.__file__)==CORE_SHA,'Frozen receiver core changed')
    # Preserve the original complete-grid loop and scientific functions. Add one
    # primed normal-frame cache entry and count its first logical occurrence once.
    tree=ast.parse(inspect.getsource(original.images));changed=dict(prime=0,reuse=0)
    class PrimeNormalFrame(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='cache' and isinstance(node.value,ast.Dict):
                changed['prime']+=1
                return ast.parse('cache,primed_key=prime(r,backend,codec,partial,ledger,boundary,out,g,read)\nprimed_seen=False').body
            return self.generic_visit(node)
        def visit_If(self,node):
            self.generic_visit(node)
            if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='key' and \
                    len(node.test.ops)==1 and isinstance(node.test.ops[0],ast.NotIn) and isinstance(node.test.comparators[0],ast.Name) and node.test.comparators[0].id=='cache':
                changed['reuse']+=1
                node.orelse=ast.parse("if key==primed_key and not primed_seen:\n primed_seen=True;reuse['actual_new_inputs']+=1;mode='ACTUAL_NEW_PRIMED_NORMAL_FRAME'\nelse:\n reuse['exact_same_input']+=1;mode='EXACT_ACTUAL_INPUT_REUSE'").body
            return node
    tree=PrimeNormalFrame().visit(tree);require(changed==dict(prime=1,reuse=1),'Original cache loop structure changed')
    ns=dict(vars(original));ns.update(prime=prime)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),ns)
    result=ns['images'](r,backend,codec,partial,ledger,boundary,out,g,read)
    result['device_relocation_witness']=original.pilot.descriptor(Path(out)/'relocation_witness_check.json')
    result['extra_qualification_calls']=0
    return result
