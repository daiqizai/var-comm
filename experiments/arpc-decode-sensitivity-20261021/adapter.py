"""Change only untransmitted-scale sampling in a private function namespace."""
import ast
import copy
import hashlib

NATIVE_SHA = '29cf445d424c7f26bf8578ded1ba30354c81846f44a18abeb4e3f1dc0074a03c'
SETTINGS = ('original', 'greedy', 'low_tau025')


def temperatures(setting, endpoint):
    assert setting in SETTINGS and 1 <= endpoint <= 7
    return [0.5] * endpoint + [0.25 if setting == 'low_tau025' else 0.5] * (7-endpoint)


def choose(logits, setting, si, received_endpoint, original_sampler, **kwargs):
    assert setting in SETTINGS
    if setting == 'greedy' and si >= received_endpoint:
        # An explicit maximum, with first-index tie breaking; no multinomial call.
        return logits.argmax(dim=-1, keepdim=True)
    return original_sampler(logits, **kwargs)


def generation_function(receiver, setting, source):
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decompress_cfg')
    changed = copy.deepcopy(original)
    calls = []
    for n in ast.walk(changed):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'sample_with_top_k_top_p_also_inplace_modifying_logits_':
            calls.append(n)
            n.func.id = '_selected_sampler'
            n.keywords += [ast.keyword(arg='si',value=ast.Name(id='si',ctx=ast.Load())),
                          ast.keyword(arg='received_endpoint',value=ast.parse('gt_leak+1',mode='eval').body)]
    assert len(calls) == 2
    inverse = copy.deepcopy(changed)
    for n in ast.walk(inverse):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == '_selected_sampler':
            n.func.id = 'sample_with_top_k_top_p_also_inplace_modifying_logits_'
            n.keywords = n.keywords[:-2]
    assert ast.dump(inverse) == ast.dump(original)
    native_sampler = receiver.sample_with_top_k_top_p_also_inplace_modifying_logits_
    def selected(logits, si, received_endpoint, **kw):
        return choose(logits,setting,si,received_endpoint,native_sampler,**kw)
    ns = dict(vars(receiver)); ns['_selected_sampler'] = selected
    module=ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))
    exec(compile(module,'ARPC::generation_only_sensitivity','exec'),ns)
    return ns['decompress_cfg'], hashlib.sha256(ast.dump(module).encode()).hexdigest()


def native_program(data):
    assert hashlib.sha256(data).hexdigest() == NATIVE_SHA
    text=data.decode()
    # Reuse the exact qualified loading, numerical settings, memory guard and
    # image saving code. No TX or source decoding branch is executed.
    text=text[:text.index('    result=[]\n')]
    text=text.replace("'rx_prefix':24,'rx_render':24", "'rx_prefix':0,'rx_render':cfg['render_cap']")
    text=text.replace("out,pathlib.Path(cfg['run'])/'wire_inputs'", "out,pathlib.Path(cfg['run'])/'wire_inputs',cfg['sensitivity_code']")
    old='receiver.decompress_cfg(prior,vae,schedule,caption,text_tokenizer,text_encoder,e-1,tokens,cfg_list=3,tau_list=0.5)'
    new="generation(prior,vae,schedule,caption,text_tokenizer,text_encoder,e-1,tokens,cfg_list=[3]*7,tau_list=temperatures(setting,e))"
    assert text.count(old)==1
    text=text.replace(old,new)
    text=text.replace("    gpu=torch.cuda.get_device_properties(0)", """    assert torch.__version__=='2.10.0+cu130' and sys.version_info[:2]==(3,12)
    import ctypes
    lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();bus=ctypes.create_string_buffer(32)
    assert lib.cuDeviceGet(ctypes.byref(dev),0)==0
    assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0
    assert bus.value.decode().lower()==cfg['resource']['gpu']['pci_bus_id'].lower()
    gpu=torch.cuda.get_device_properties(0)""")
    return text
