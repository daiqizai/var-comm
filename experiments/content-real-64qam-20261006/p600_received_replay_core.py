"""Prepared P600 received-Z replay core. No CLI, loader construction or launch.

The future registered owner supplies the original native runtime, the independent
classifier and exact source pixels. This core never sends a waveform, generates
noise, runs the P receiver, selects a policy or writes a packet ledger.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

SEEDS = (2001, 2002, 2003)
SNRS = (13, 19)
DC_STATE = 'bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1'
CONVNEXT = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
FLAGS = dict(matmul_tf32=False, cudnn_tf32=False, precision='highest', cudnn_benchmark=False,
             cudnn_deterministic=True, deterministic=True, threads=6, interop_threads=2)
SCOPE = 'P600_EXISTING_RECEIVED_LATENT_RGB_AND_INDEPENDENT_CONVNEXT_ONLY'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def raw_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rgb(array):
    a = np.asarray(array)
    require(a.dtype == np.float32 and a.shape == (3,256,256) and np.isfinite(a).all()
            and a.min() >= 0 and a.max() <= 1, 'Original finite float32 CHW RGB01 required; no clamp')
    return np.ascontiguousarray(a)


def rgb_sha(array):
    # Exact original replay.rgb_fingerprint domain, not the H tensor SHA domain.
    return hashlib.sha256(b'float32:3,256,256:RGB\0'+rgb(array).tobytes()).hexdigest()


def row_key(row):
    return tuple(int(row[k]) for k in ('source_index','snr_db','noise_seed'))


def context(completion, inventory, scalar_inventory, metrics_registration,
            native_qualification, *, inventory_sha256):
    """Pure document admission; caller must verify actual file pins first."""
    require(completion['status'] == 'P600_RECEIVED_LATENT_AUDIT_COMPLETE'
            and completion['source_count'] == 100 and completion['frame_count'] == 600
            and list(completion['outputs'].values()) == [inventory_sha256], 'Actual CPU latent audit required')
    require(inventory['status'] == 'P600_RECEIVED_LATENT_METADATA_VERIFIED'
            and inventory['input_bindings'] == completion['input_bindings']
            and inventory['source_bindings'] == completion['source_bindings']
            and inventory['Z_shape'] == [1500,32,16,16] and inventory['Z_dtype'] == 'float32'
            and inventory['source_indices_shape'] == [1500] and inventory['source_indices_dtype'] == 'int64'
            and inventory['original_grid_frames'] == 1500 and inventory['selected_frames'] == 600
            and inventory['selected_snrs'] == list(SNRS) and inventory['noise_seeds'] == list(SEEDS),
            'Original selected latent schema differs')
    require(all(completion[k] is False and inventory[k] is False for k in
            ('GPU_used','models_constructed','tensor_values_exported','RGB_parity_verified','ConvNeXt_scored','full_P_reuse_complete'))
            and all(completion[k] == inventory[k] == 0 for k in ('new_noise_draws','new_packet_decodes','budget_writes')),
            'Latent audit is not the original metadata-only predecessor')
    replay = metrics_registration['original_replay_inventory']
    require(metrics_registration['numerical_runtime'] == FLAGS
            and native_qualification['status'] == 'REAL_NATIVE_QUALIFICATION_PASS'
            and native_qualification['frozen_identity'] == replay['model_identity']
            and all(native_qualification['numerical_runtime'][k] == v for k,v in FLAGS.items())
            and inventory['Dc_state_sha256'] == replay['model_identity']['models']['decoder'] == DC_STATE,
            'Original numerical/visual identity differs')
    ids = inventory['identity']['source_ids']; pre = inventory['identity']['preprocessing_ids']
    require(ids == replay['source_ids'] == metrics_registration['source_ids']
            and pre == replay['preprocessing_ids'] and len(ids) == len(set(ids)) == len(pre) == 100
            and inventory['identity']['role'] == 'development' and inventory['identity']['seeds'] == list(SEEDS),
            'Original source100/noise identity differs')
    wanted = set(itertools.product(range(100),SNRS,SEEDS))
    selected, scalar = {}, {}
    for rows, target in ((inventory['selected_rows'],selected),(scalar_inventory['rows'],scalar)):
        require(len(rows) == 600, 'Exactly600 rows required')
        for r in rows:
            key = row_key(r)
            require(key in wanted and key not in target, 'Duplicate/missing/unexpected selected frame')
            require(r['source_id'] == ids[key[0]] and r['preprocessing_id'] == pre[key[0]], 'Source identity mismatch')
            target[key] = copy.deepcopy(r)
    require(set(selected) == set(scalar) == wanted, 'Selected grid incomplete')
    indices = set()
    for key, r in selected.items():
        s = scalar[key]; j = r['original_tensor_index']
        require(type(j) is int and 0 <= j < 1500 and j not in indices, 'Received tensor index alias or outside grid')
        indices.add(j)
        require(s['method'] == 'P1024' and s['model_id'] == s['base_model_id'] == 'P1024_step40000_12b979260ebf'
                and s['decoder_id'] == s['replay_decoder_id'] == 'Dc' and int(s['N']) == 1024
                and str(s['label_conditioned']).lower() == 'false'
                and s['image_sha256'] == r['existing_image_sha256']
                and s['reference_sha256'] == r['existing_reference_sha256']
                and s['waveform_sha256'] == r['waveform_sha256']
                and s['observation_sha256'] == r['observation_sha256'], 'Original scalar provenance differs')
        require(not any(k.startswith('convnext_') or k.startswith('p_replay_') for k in s),
                'New metric fields collide with original scalar record')
        require(row_key(inventory['channel_rows'][j]) == key, 'Selected index points to another observation')
    policy_pins = [v for p,v in inventory['input_bindings'].items() if p.endswith('/selected_P1024.json')]
    require(len(policy_pins)==1 and isinstance(policy_pins[0],str) and len(policy_pins[0])==64,
            'One original selected P policy pin required before model admission')
    return dict(scope=SCOPE, source_ids=copy.deepcopy(ids), preprocessing_ids=copy.deepcopy(pre),
        inventory=copy.deepcopy(inventory), selected=selected, scalar=scalar,
        native_identity=copy.deepcopy(replay['model_identity']), native_runtime=copy.deepcopy(native_qualification['numerical_runtime']),
        flags=copy.deepcopy(FLAGS), selected_P_sha256=policy_pins[0])


class SealedLatents:
    """Read the pinned tensor on CPU only; injected Torch is already qualified."""
    def __init__(self, ctx, torch):
        inv = ctx['inventory']; path = Path(inv['original_tensor_path'])
        require(path.is_file() and sha(path) == inv['original_tensor_sha256']
                == inv['input_bindings'][str(path)], 'Sealed received tensor changed')
        data = torch.load(path, map_location='cpu', weights_only=True)
        require(type(data) is dict and set(data) == {'Z','source_indices','rows','identity'}, 'Unknown tensor schema')
        require(type(data['Z']) is torch.Tensor and type(data['source_indices']) is torch.Tensor
                and data['Z'].device.type == data['source_indices'].device.type == 'cpu'
                and data['Z'].dtype == torch.float32 and data['source_indices'].dtype == torch.int64,
                'Original CPU tensor dtype required')
        require(data['identity'] == inv['identity'] and data['rows'] == inv['channel_rows'], 'Original tensor metadata changed')
        self.z = data['Z'].numpy(); indices = data['source_indices'].numpy()
        require(self.z.shape == (1500,32,16,16) and np.isfinite(self.z).all()
                and indices.shape == (1500,) and all(int(indices[j]) == r['source_index']
                    for j,r in enumerate(data['rows'])), 'Original latent grid invalid')
        self.ctx, self.path = ctx, path
        require(sha(path) == inv['original_tensor_sha256'], 'Tensor changed while loading')

    def batch(self, index, snr):
        return latent_batch(self.ctx, self.z, index, snr)


def latent_batch(ctx, z, index, snr):
    require(type(index) is int and 0 <= index < 100 and snr in SNRS, 'Only original selected source/SNR')
    records = [ctx['selected'][index,snr,n] for n in SEEDS]
    values = []
    for record in records:
        a = np.asarray(z[record['original_tensor_index']])
        require(a.dtype == np.float32 and a.shape == (32,16,16) and np.isfinite(a).all()
                and raw_sha(a) == record['Z_sha256'], 'Selected received Z byte identity differs')
        values.append(a)
    return np.stack(values)  # Actual original three-noise batch, no batch-size search.


def runtime_flags(torch):
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        precision=torch.get_float32_matmul_precision(),cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic=torch.are_deterministic_algorithms_enabled(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())


class FrozenDc:
    """Use already-loaded qualified Dc directly. Never construct P/VAR or call a channel."""
    def __init__(self, native, ctx):
        self.native, self.ctx, self.torch = native, ctx, native.torch
        require(native.loaded['identity'] == ctx['native_identity']
                and str(native.loaded['device']) == 'cuda:0'
                and native.torch.__version__ == ctx['native_runtime']['torch']
                and native.torch.version.cuda == ctx['native_runtime']['cuda'], 'Original native runtime differs')
        native.frozen()
        self.check()

    def check(self):
        decoder = self.native.loaded['decoder']
        require(runtime_flags(self.torch) == self.ctx['flags'] and not decoder.training
                and not any(p.requires_grad or p.grad is not None for p in decoder.parameters()),
                'Frozen Dc numerical/runtime flags changed')

    def __call__(self, z):
        self.check()
        require(type(z) is np.ndarray and z.shape == (3,32,16,16) and z.dtype == np.float32,
                'Original B3 received latent required')
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(np.ascontiguousarray(z)).to(self.native.loaded['device'])
            result = self.native.loaded['decoder'](x)
            require(result.dtype == t.float32 and tuple(result.shape) == (3,3,256,256), 'Original Dc output shape/dtype differs')
            output = result.detach().cpu().numpy().copy()
        self.check()
        return output

    def finish(self):
        self.check()
        self.native.frozen()


def classifier_check(classifier, ctx):
    v = classifier.identity
    require(v['weights_sha256'] == CONVNEXT and v['classification_batch_size'] == 1
            and v['used_for_selection'] is False and v['population'] == 'development'
            and v['inference_precision'] == 'float32' and v['policy_sha256'] == ctx['selected_P_sha256']
            and isinstance(ctx['selected_P_sha256'],str) and len(ctx['selected_P_sha256']) == 64,
            'Original independent classifier identity/admission differs')
    return identity(v)


def replay_source(ctx, index, pixels, *, latents, decoder, classifier, guard=lambda:None):
    """Return six original float images plus original scalar rows and new classifier fields.

    Decoder inputs contain only received Z. Source truth is used for hashes and
    independent evaluation, and never enters the decoder or repairs its output.
    """
    require(type(index) is int and 0 <= index < 100, 'Original source index required')
    require(type(pixels) is np.ndarray and pixels.shape == (3,256,256) and pixels.dtype == np.uint8
            and raw_sha(pixels) == ctx['preprocessing_ids'][index], 'Original uint8 source pixels differ')
    target = np.ascontiguousarray(pixels.astype(np.float32)/np.float32(255.))
    target_hash = rgb_sha(target)
    records = [ctx['selected'][index,s,n] for s in SNRS for n in SEEDS]
    require(all(r['existing_reference_sha256'] == target_hash for r in records), 'Original reference RGB hash differs')
    class_identity = classifier_check(classifier,ctx)
    images = {}
    # Verify every reconstructed pixel hash before computing any classifier score.
    for snr in SNRS:
        guard()
        batch = latents.batch(index,snr)
        require(batch.dtype == np.float32 and batch.shape == (3,32,16,16), 'Received batch differs')
        require(all(raw_sha(batch[j]) == ctx['selected'][index,snr,n]['Z_sha256']
                    for j,n in enumerate(SEEDS)), 'Decoder input is not the exact received latent')
        outputs = decoder(batch.copy())
        require(type(outputs) is np.ndarray and outputs.dtype == np.float32
                and outputs.shape == (3,3,256,256), 'Dc B3 output differs')
        for j,seed in enumerate(SEEDS):
            image = rgb(outputs[j]).copy()
            require(rgb_sha(image) == ctx['selected'][index,snr,seed]['existing_image_sha256'],
                    'P RGB differs from sealed old scalar image: diagnose, never overwrite')
            images[snr,seed] = image
    labels = {int(ctx['scalar'][index,s,n]['true_class_index']) for s in SNRS for n in SEEDS}
    require(len(labels) == 1 and 0 <= next(iter(labels)) < 1000, 'Original ImageNet label differs')
    label = next(iter(labels)); guard(); source_prediction = classifier.predict(target)
    require(type(source_prediction) is int and 0 <= source_prediction < 1000, 'Invalid classifier source prediction')
    rows = []
    for snr in SNRS:
        for seed in SEEDS:
            guard(); key = index,snr,seed; old = ctx['scalar'][key]; received = ctx['selected'][key]
            scored = classifier.score(images[snr,seed],true_label=label,source_prediction=source_prediction)
            require(type(scored) is dict and not set(scored)&set(old)
                    and scored['convnext_true_class'] == label
                    and scored['convnext_source_prediction'] == source_prediction
                    and type(scored['convnext_prediction']) is int and 0 <= scored['convnext_prediction'] < 1000
                    and scored['convnext_used_for_selection'] is False
                    and scored['convnext_weight_sha256'] == CONVNEXT, 'Classifier output/provenance differs')
            pred = scored['convnext_prediction']
            require(scored['convnext_top1_label'] == (pred==label)
                    and scored['convnext_source_prediction_agreement'] == int(pred==source_prediction),
                    'Classifier accuracy fields disagree with predictions')
            row = copy.deepcopy(old)
            row.update(scored)
            row.update(p_replay_original_scalar_sha256=identity(old),p_replay_exact_RGB=True,
                p_replay_original_tensor_index=received['original_tensor_index'],p_replay_received_Z_sha256=received['Z_sha256'],
                p_replay_classifier_identity=class_identity,p_replay_new_noise_draws=0,p_replay_new_packet_decodes=0)
            rows.append(row)
    require(classifier_check(classifier,ctx) == class_identity, 'Classifier identity changed during source')
    return dict(source_index=index,source_id=ctx['source_ids'][index],rows=rows,images=images,
        reference_sha256=target_hash,RGB_parity_frames=6,ConvNeXt_frames=6,source_predictions=1,
        new_noise_draws=0,new_packet_decodes=0,policy_selection=False)


def validate_complete(ctx, rows):
    """600-row completion gate for a future writer; no aggregate metric selection."""
    require(type(rows) is list and len(rows) == 600, 'Exactly600 final P rows required')
    seen = set()
    for row in rows:
        key = row_key(row)
        require(key in ctx['scalar'] and key not in seen, 'Final P grid missing or duplicate')
        seen.add(key); old=ctx['scalar'][key]
        require(all(row.get(k)==v for k,v in old.items())
                and row['p_replay_original_scalar_sha256']==identity(old)
                and row['p_replay_exact_RGB'] is True and row['p_replay_new_noise_draws']==0
                and row['p_replay_new_packet_decodes']==0 and row['convnext_used_for_selection'] is False,
                'Original scalar overwritten or unverified P output')
        received=ctx['selected'][key]
        require(row['p_replay_original_tensor_index']==received['original_tensor_index']
                and row['p_replay_received_Z_sha256']==received['Z_sha256']
                and row['convnext_weight_sha256']==CONVNEXT
                and row['convnext_true_class']==int(old['true_class_index'])
                and type(row['convnext_prediction']) is int and 0<=row['convnext_prediction']<1000
                and type(row['convnext_source_prediction']) is int and 0<=row['convnext_source_prediction']<1000
                and row['convnext_top1_label']==(row['convnext_prediction']==row['convnext_true_class'])
                and row['convnext_source_prediction_agreement']==int(row['convnext_prediction']==row['convnext_source_prediction']),
                'Completed P classifier/latent proof differs')
    require(seen == set(ctx['scalar']), 'Incomplete P final grid')
    return dict(status='P600_CORE_OUTPUT_VALIDATED_NOT_OWNER_COMPLETION',source_count=100,frame_count=600,
        noise_seeds=list(SEEDS),snrs_db=list(SNRS),RGB_parity_frames=600,ConvNeXt_frames=600,
        original_scalar_values_preserved=True,new_noise_draws=0,new_packet_decodes=0,policy_selection=False)
