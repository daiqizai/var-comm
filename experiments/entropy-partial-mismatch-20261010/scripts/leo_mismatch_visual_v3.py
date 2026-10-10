"""One explicit GPU5 recovery of raw mismatch reconstruction on original leo.

The failed H800 initialization remains spent and unresolved. This registration
allows one further model-load attempt (two cumulative), without another PHY or
any additional reconstruction/prior budget. Original resource limits persist.
"""
from __future__ import annotations
import argparse
import ast
import inspect
import os
from pathlib import Path
import time
import types
import leo_whole_gate_v1 as original_gate
import leo_single_rx_diagnostic_v2 as original_device
import h800_mismatch_visual_v1 as base

BASE_SHA='de547876187cf8e69a68b964f8e8170a0326ea126355e85d583640a89d15e59b'
NATIVE_SHA='1d2cfe5cf87e4961a40d6cd7bdc2448dd2d4c24edb54824279a4ae6172f449ec'
RT=original_gate.BASE/'var_comm_runtime_20261010'
NATIVE_PATH=RT/'qualification/leo_raw_mismatch_native_recheck_v1/completion.json'
FAILED_ROOT=RT/'qualification/h800_mismatch_visual_v2_attempt1'
OUT=RT/'qualification/leo_mismatch_visual_v3_attempt1/registered/run'
FAILURE_PINS={
 'registered/request.json':'9fa7fc947c780449f4f873546a52e793babd660476fe76d23710ec56485b3f31',
 'run_owner_actual_wait.json':'4d2579bfa3bea1139084ba841393cbbbf76d3a48bf67368d507bd73996592b5c',
 'registered/run/actual_child_wait.json':'0923eff6767843d1cc6bf76f9f3a0661baef44acea81e0f3b3ad5ab1a7c8696a',
 'registered/run/worker_failure.json':'feba7c32fcad38c02b084be6561c0b1841a15ac8b2c0f434ba613940c583594d',
 'registered/run/calls/0001_model_load.reserved.json':'8a27b6c82d5816d04e683c0e89ff7c1fde2e987fde873bb18d8160d6079a1b58'}
SCHEMA='LEO_RAW100_ONE_BIN_MISMATCH_VISUAL_V3_GPU5'
PASS='PASS_LEO_RAW100_MISMATCH_RECONSTRUCTIONS_ONLY_GPU5'
CAPS=dict(base.CAPS)
TOOLS=base.TOOLS+('h800_mismatch_visual_v1.py',)
gate=base.gate;raw=base.raw;physical=base.physical;execution=base.execution


class LeoGate:
    def __getattr__(self,name):return getattr(original_gate,name)
    def native_pin(self):return dict(path=str(NATIVE_PATH),sha256=NATIVE_SHA)
    def validate_spec(self,spec,base=original_gate.BASE,verify_environment=True):
        receipt=original_gate.checked_json(NATIVE_PATH,NATIVE_SHA)
        original_gate.require(receipt['status']=='ORIGINAL_LEO_BYTES_REVERIFIED_NO_NUMERICAL_CALL' and
            receipt['CUDA_VISIBLE_DEVICES']=='' and receipt['model_calls']==receipt['tensor_calls']==0 and
            receipt['original_PFS_runtime_rows_unchanged'] is True,'Actual original leo runtime recheck required')
        result=original_gate.validate_spec(spec,base,verify_environment)
        original_gate.require(result[1].host_native==receipt['actual_native_files'] and
            spec['environment']==receipt['original_projection'] and spec['host_native_addendum']==receipt['original_host_native_addendum'],
            'Current leo original native binding differs')
        return result


g=LeoGate()


class DeviceAdapter(original_device.DeviceAdapter):
    def __init__(self,pin,index):
        g.require(type(index) is int and index==5 and pin==g.native_pin(),'Only independently rechecked original leo GPU5 required')
        receipt=g.checked_json(NATIVE_PATH,NATIVE_SHA)
        g.require(receipt['registered_device_identity']==original_gate.device_identity_binding(),'Original leo device registration differs')
        self.index=5;self.UUID=g.UUID;self.CUDA_UUID=g.CUDA_UUID;self.PCI_BUS_ID=g.PCI_BUS_ID
    def __getattr__(self,name):return getattr(g,name)
    def resource_snapshot(self,shared,prelaunch=False):
        try:return super().resource_snapshot(shared,prelaunch)
        except original_device.ResourceBusy as error:
            # Keep the original thresholds and preserve the failed observation.
            if hasattr(error,'resource_snapshot') and OUT.is_dir():
                dest=g.inside(OUT/'resource_refusals');dest.mkdir(exist_ok=True)
                g.write(dest/f'{os.getpid()}-{time.time_ns()}.json',dict(error=str(error),snapshot=error.resource_snapshot,
                    pid=os.getpid(),prelaunch=prelaunch,thresholds_unchanged=True))
            raise


def failed_attempt():
    values={rel:g.checked_json(FAILED_ROOT/rel,digest) for rel,digest in FAILURE_PINS.items()}
    owner=values['run_owner_actual_wait.json'];child=values['registered/run/actual_child_wait.json']
    failure=values['registered/run/worker_failure.json'];counts=failure['counts'];request=values['registered/request.json']
    expected=dict.fromkeys(CAPS,0);reserved=dict(expected,model_load=1)
    g.require(owner['actual_wait'] is True and owner['returncode']==1 and owner['timeout'] is False and
        child['actual_child_waited'] is True and child['child_exit_code']==1 and child['success'] is False and
        counts['caps']==CAPS and counts['reserved']==reserved and counts['completed']==expected and counts['unresolved']==1 and
        failure['request_sha256']==FAILURE_PINS['registered/request.json'] and
        failure['error']=="ResourceBusy('Less than20GiB GPU margin')",'Exact paid initialization failure required')
    g.require(values['registered/run/calls/0001_model_load.reserved.json']['kind']=='model_load' and
        request['historical_image_reuse_admitted']==0,'Failed initialization ledger changed')
    policy=dict(failed_attempt_pins={k:dict(path=str(FAILED_ROOT/k),sha256=v) for k,v in FAILURE_PINS.items()},
        previous_model_load_attempts=1,previous_model_load_completed=0,previous_unresolved_model_load=1,
        additional_model_load_attempt_cap=1,cumulative_model_load_attempt_cap=2,prior_failed_ledger_preserved=True,
        previous_render_calls=0,previous_prior_calls=0,previous_decoder_calls=0,new_PHY_calls=0,
        cumulative_render_cap=4500,cumulative_prior_cap=45000,cumulative_decoder_cap=4500,
        automatic_retry=False,historical_leo_GPU7_SIGKILL_cause='UNDETERMINED')
    return request,policy


def visual_identity(device):
    return dict(schema='LEO_RAW_KEEP_RECONSTRUCTION_IDENTITY_V3',device_identity=device,native_binding=g.native_pin(),
        model_state_identity=dict(g.MODELS),numeric_settings=base.visual_identity(device)['numeric_settings'],
        source_math_SHA=base.visual_identity(device)['source_math_SHA'],
        render_source_SHA=g.CODE['experiments/content-real-64qam-20261006/h_source_driver.py'],
        receiver_semantics='Original raw full433 hard-token KEEP, unconditional argmax VAR completion, frozen Dc',
        old_host_images_admitted=False,H800_numerical_equivalence_claimed=False)


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    previous,recovery=failed_attempt();cpu,worker,logical=base.cpu_closure(r['CPU_closed'])
    identity=visual_identity(runner.device_from_request(r).device_identity_binding())
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and r['execution']==e and
        r['target_index']==5 and r['recovery_budget']==recovery and r['physical_frames']==worker['results']['physical_frames'] and
        r['logical_frames']==logical and r['raw_runtime_binding']==cpu['raw_runtime_binding'] and r['visual_identity']==identity and
        r['spec']==runner.inherited_spec(previous,e['deadline_unix'],900),'Frozen independent leo raw request changed')
    g.require(r['source_TX_calls']==r['source_RX_calls']==r['PHY_calls']==r['metric_model_calls']==r['Direct_calls']==0 and
        r['automatic_successor'] is False and r['historical_image_reuse_admitted']==0,'Raw VAR-only scope changed')
    runner.check_tools(r['tool_bindings']);return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(out==OUT.parent and not out.exists(),'Exact fresh leo GPU5 registration required')
    pins=dict(completion=dict(path=str(g.inside(a.cpu_completion)),sha256=a.cpu_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.cpu_owner_wait)),sha256=a.cpu_owner_wait_sha256))
    cpu,worker,logical=base.cpu_closure(pins);previous,recovery=failed_attempt();e=execution(a.deadline_unix,a.max_seconds)
    g.require(pins==previous['CPU_closed'] and a.target_index==5,'Same actual raw packets and original GPU5 required')
    spec=runner.inherited_spec(previous,e['deadline_unix'],900);g.validate_spec(spec)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=DeviceAdapter(dp,a.target_index)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,CPU_closed=pins,recovery_budget=recovery,
        physical_frames=worker['results']['physical_frames'],logical_frames=logical,raw_runtime_binding=cpu['raw_runtime_binding'],
        visual_identity=visual_identity(device.device_identity_binding()),device_receipt=dp,target_index=5,
        device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        source_TX_calls=0,source_RX_calls=0,PHY_calls=0,metric_model_calls=0,Direct_calls=0,
        historical_image_reuse_admitted=0,automatic_successor=False,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)})
    out.mkdir(parents=True);path=out/'request.json';g.write(path,r);return raw.pin(path)


def run(a,runner):
    tree=ast.parse(inspect.getsource(base.original.run))
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value==43200]
    g.require(len(matches)==1,'Original finite visual owner count guard changed');matches[0].value=5400
    ns=dict(base.original.run.__globals__);ns.update(g=g,__file__=__file__,registration=registration,PASS=PASS,SCHEMA=SCHEMA)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),ns)
    return ns['run'](a,runner)


def engine():
    g.require(g.sha(base.__file__)==BASE_SHA,'Frozen raw reconstruction implementation changed')
    ns=dict(vars(base.engine()));ns.update(g=g,DeviceAdapter=DeviceAdapter,__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


ns=dict(vars(base));ns.update(g=g,__file__=__file__,registration=registration,visual_identity=visual_identity,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS)
for name in ('parser','received_tokens','actual_state_key','render_actual','images','worker'):
    ns[name]=gate.source.host.clone_function(getattr(base,name),ns);globals()[name]=ns[name]


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare')
    for name in ('cpu-completion','cpu-completion-sha256','cpu-owner-wait','cpu-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(5,),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine();result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
