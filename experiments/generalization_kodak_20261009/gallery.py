"""Render prespecified Kodak panels from completed float-RGB caches only.

No model, metric, channel, policy selection, bootstrap or quality ranking.
The protocol fixes source indices 0..3, noise seed 2001 and SNR 4/10/19.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys

METHODS = ('SWIN80K', 'P1024', 'BPG_ADAPTIVE', 'VAR_UNCONDITIONAL')
LABELS = {
    'ORIGINAL': 'Original image',
    'SWIN80K': 'SwinJSCC-80k\n(adapted)',
    'P1024': 'Latent continuous\nJSCC',
    'BPG_ADAPTIVE': 'Adaptive BPG\n+ LDPC',
    'VAR_UNCONDITIONAL': 'Partial digital transmission\n+ unconditional VAR\n(ours)',
}
SNRS = (4, 10, 19)
SOURCES = (0, 1, 2, 3)
SEED = 2001
PREPROCESSING = 'kodak_rgb_center_crop_256_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def descriptor(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def rgb_hash(image):
    import numpy as np
    value = np.ascontiguousarray(image)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def validate_rgb(value):
    import numpy as np
    require(value.dtype == np.float32 and value.shape == (3, 256, 256)
            and np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1,
            'Expected unchanged float32 CHW RGB in [0,1]')
    return np.ascontiguousarray(value)


def load_sources(manifest_path, protocol_path):
    import numpy as np
    from PIL import Image
    manifest, protocol = read(manifest_path), read(protocol_path)
    require(protocol['schema'] == 'KODAK24_FROZEN_GENERALIZATION_PROTOCOL_V1'
            and protocol['dataset_manifest_sha256'] == sha(manifest_path)
            and protocol['N_complex_channel_uses'] == 1024
            and protocol['prespecified_display'] == dict(source_indices=list(SOURCES), noise_seed=SEED, snr_db=list(SNRS)),
            'Use the actual pre-inference display protocol; no gallery reselection')
    require(manifest['schema'] == 'KODAK24_CENTER256_DATASET_V1' and manifest['source_count'] == 24
            and manifest['preprocessing'] == PREPROCESSING, 'Wrong Kodak center-crop population')
    records = manifest['records']
    require(len(records) == 24 and [r['source_index'] for r in records] == list(range(24))
            and [r['source_id'] for r in records] == [f'kodak/kodim{i+1:02d}' for i in range(24)],
            'Require the complete fixed Kodak order')
    images = {}
    for index in SOURCES:
        record = records[index]
        require(record['preprocessing_id'] == PREPROCESSING
                and sha(record['png']['path']) == record['png']['sha256'], 'Original crop binding changed')
        with Image.open(record['png']['path']) as image:
            require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256), 'Unmodified original RGB256 PNG required')
            pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == record['pixels_chw_uint8_sha256'], 'Original crop pixel hash changed')
        images[index] = validate_rgb(pixels.astype(np.float32) / np.float32(255))
    return records, images


def load_method(method, frames_path, records):
    """Require all 216 completed frames, then read only 12 preselected images."""
    import numpy as np
    frames_path = Path(frames_path).resolve()
    completion_path = frames_path.parent / 'completion.json'
    completion, frames = read(completion_path), read(frames_path)
    expected_status = 'KODAK24_ADAPTIVE_BPG_ACTUAL_CHILDREN_WAIT_ZERO' if method == 'BPG_ADAPTIVE' else 'COMPLETE'
    require(completion['status'] == expected_status and completion['source_count'] == 24
            and completion.get('frame_count', completion.get('frames')) == 216
            and completion['outputs'].get(str(frames_path)) == sha(frames_path),
            'Method frames lack a matching completion receipt: ' + method)
    if method == 'BPG_ADAPTIVE':
        require(completion['actual_children_waited'] and completion['worker_exit_codes'] == [0, 0],
                'Both BPG children must have actually completed')
    else:
        require(completion['method'] == method, 'Wrong neural completion identity')
    require(isinstance(frames, list) and len(frames) == 216, 'Complete method frame list required')
    expected = {(i, s, n) for i in range(24) for s in SNRS for n in (2001, 2002, 2003)}
    seen, selected = set(), {}
    for frame in frames:
        key = frame['source_index'], frame['snr_db'], frame['noise_seed']
        require(key in expected and key not in seen and frame['method'] == method
                and frame['source_id'] == records[key[0]]['source_id']
                and frame.get('N', 1024) == 1024 and isinstance(frame['status'], str) and frame['status'],
                'Duplicate or mismatched method/source/SNR/noise frame')
        seen.add(key)
        if key[0] not in SOURCES or key[2] != SEED:
            continue
        archive = frame['reconstruction']
        require(archive['key'] == 'rgb' and sha(archive['path']) == archive['sha256'], 'Actual reconstruction archive changed')
        with np.load(archive['path'], allow_pickle=False) as content:
            image = validate_rgb(content['rgb'].copy())
        selected[key[0], key[1]] = dict(frame=frame, image=image)
    require(seen == expected and len(selected) == 12, 'Missing frames; never filter failed or unattractive outputs')
    return selected, dict(frames=descriptor(frames_path), completion=descriptor(completion_path))


def caption(snr):
    text = (
        f'Prespecified Kodak qualitative comparison at $N=1024$ complex channel uses and {snr} dB. '
        'Rows are Kodak images 01--04 in numeric order; columns show the original image, '
        'SwinJSCC-80k (adapted), latent continuous joint source--channel coding, '
        'adaptive BPG with LDPC, and partial digital transmission with unconditional VAR completion (ours). '
        'Each original is the fixed central $256\\times256$ RGB crop, with floor integer offsets and no resizing of the source image. '
        'All methods use the same crop and the prespecified noise label 2001; no source or noise realization was selected by reconstruction quality. '
        'Shared noise labels do not imply identical received observations across different waveforms. '
        'Reconstruction panels are read directly from completed float32 RGB caches; all method failure outputs are retained, with no sharpening or image enhancement. '
        'VAR uses the null class embedding and no supplied source class. '
    )
    if snr == 19:
        text += "19 dB is outside Swin's training and calibration range. "
    return text.strip()


def render(snr, images, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'text.usetex': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(4, 5, figsize=(7.2, 6.24), facecolor='white')
    fig.subplots_adjust(left=.047, right=.993, bottom=.038, top=.867, wspace=.035, hspace=.035)
    columns = ('ORIGINAL', *METHODS)
    for row, source in enumerate(SOURCES):
        for col, method in enumerate(columns):
            axis = axes[row, col]
            axis.imshow(images[source, method].transpose(1, 2, 0), interpolation='none', aspect='equal')
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                title = LABELS[method]
                if method == 'SWIN80K' and snr == 19:
                    title += '*'
                axis.set_title(title, fontsize=7.1, fontweight='normal', pad=7, linespacing=1.15)
            if col == 0:
                axis.text(-.085, .5, f'Kodak {source+1:02d}', transform=axis.transAxes,
                          rotation=90, ha='center', va='center', fontsize=7.4)
    fig.suptitle(f'Kodak center crops  |  N = 1024  |  SNR = {snr} dB', x=.52, y=.979, fontsize=9.2)
    if snr == 19:
        fig.text(.52, .014, "* 19 dB is outside Swin's training and calibration range.", ha='center', fontsize=6.7)
    else:
        fig.text(.52, .014, 'Prespecified sources 01–04; fixed noise label 2001.', ha='center', fontsize=6.7)
    stem = out / f'kodak_N1024_{snr:02d}dB_fixed01_04'
    for extension in ('pdf', 'svg', 'png'):
        # Photo panels remain pixels inside PDF/SVG; labels remain vector text.
        fig.savefig(str(stem) + '.' + extension, dpi=600, facecolor='white')
    plt.close(fig)
    return {extension: descriptor(str(stem) + '.' + extension) for extension in ('pdf', 'svg', 'png')}


def run(args):
    records, sources = load_sources(args.source_manifest, args.protocol)
    paths = dict(SWIN80K=args.swin_frames, P1024=args.p_frames,
                 BPG_ADAPTIVE=args.bpg_frames, VAR_UNCONDITIONAL=args.var_frames)
    loaded, provenance = {}, {}
    for method in METHODS:
        loaded[method], provenance[method] = load_method(method, paths[method], records)
    out = Path(args.out).resolve()
    require(not out.exists() or not any(out.iterdir()), 'Use a new or empty independent gallery directory')
    out.mkdir(parents=True, exist_ok=True)
    rows, figures = [], {}
    for snr in SNRS:
        panels = {}
        for source in SOURCES:
            record = records[source]
            for column, method in enumerate(('ORIGINAL', *METHODS)):
                if method == 'ORIGINAL':
                    image, status = sources[source], 'ORIGINAL_FIXED_CENTER_CROP'
                    archive, archive_hash, key = record['png']['path'], record['png']['sha256'], ''
                else:
                    item = loaded[method][source, snr]
                    image, status = item['image'], item['frame']['status']
                    value = item['frame']['reconstruction']
                    archive, archive_hash, key = value['path'], value['sha256'], value['key']
                panels[source, method] = image
                rows.append(dict(figure=f'kodak_N1024_{snr:02d}dB_fixed01_04', row=source+1, column=column+1,
                    source_index=source, source_id=record['source_id'], N=1024, snr_db=snr, noise_seed=SEED,
                    method=method, public_label=LABELS[method].replace('\n', ' '), status=status,
                    archive=archive, archive_sha256=archive_hash, image_key=key,
                    rgb_array_sha256=rgb_hash(image), preprocessing_id=PREPROCESSING,
                    selected_by_quality=False, display_sharpening=False,
                    swin_outside_training_and_calibration_range=(method == 'SWIN80K' and snr == 19)))
        figures[str(snr)] = render(snr, panels, out)
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    (out / 'display_mapping.csv').write_text(text.getvalue(), encoding='utf-8')
    latex = ['% Fixed Kodak source indices [0,1,2,3]; noise 2001; no quality-based selection.']
    captions = []
    for snr in SNRS:
        latex += ['', f'% kodak_N1024_{snr:02d}dB_fixed01_04', '\\caption{' + caption(snr) + '}',
                  f'\\label{{fig:kodak-{snr:02d}db-fixed}}']
        captions += [f'{snr} dB', caption(snr), '']
    (out / 'captions.tex').write_text('\n'.join(latex) + '\n', encoding='utf-8')
    (out / 'captions.txt').write_text('\n'.join(captions), encoding='utf-8')
    invocation = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    readme = (
        '# Prespecified Kodak qualitative gallery\n\n'
        'Three four-row, five-column figures use the pre-inference protocol: source indices '
        '`[0,1,2,3]`, noise seed `2001`, N1024 and SNRs 4, 10 and 19 dB. '
        'The source population contains all 24 Kodak images; this gallery displays its fixed first four. '
        'No selection by image quality or success status is performed.\n\n'
        'Columns: original image; SwinJSCC-80k (adapted); latent continuous JSCC; adaptive BPG + LDPC; '
        'partial digital transmission + unconditional VAR (ours). The exact internal mapping and '
        'source/archive/array hashes are retained in `display_mapping.csv`.\n\n'
        'Inputs are the same fixed central 256×256 RGB crops, with no source resizing. '
        'The renderer reads completed float32 RGB directly, without sharpening, color adjustment or '
        'failure substitution. The images are scaled only for page layout. Swin at 19 dB is outside its '
        'training and calibration range. Matching noise labels do not claim the same received waveform.\n\n'
        'Each figure is exported as 600 dpi PNG, PDF and SVG. Photo panels are embedded pixels in '
        'the PDF/SVG; text remains vector/editable where supported. `captions.tex` contains the three '
        'English captions. This script performs no metric scoring, channel simulation or inference.\n\n'
        'Reproduce with the command below, changing `--out` to a new or empty directory:\n\n```sh\n'
        + invocation + '\n```\n\n'
        'Full frozen input identities and generated output hashes are recorded in `completion.json`. '
        'Rendering completion alone does not certify manual visual inspection; inspect the PNG files before publication.\n')
    (out / 'README.md').write_text(readme, encoding='utf-8')
    outputs = {str(p): sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / 'completion.json', dict(status='KODAK24_PRESPECIFIED_GALLERY_RENDERED_V1',
        script=descriptor(__file__), protocol=descriptor(args.protocol), source_manifest=descriptor(args.source_manifest),
        method_inputs=provenance, source_indices=list(SOURCES), noise_seed=SEED, SNRs=list(SNRS),
        N=1024, figures=figures, mapping_rows=60, figure_count=3, columns=5, source_rows_per_figure=4,
        image_enhancement=False, quality_based_selection=False, new_model_calls=0,
        new_metric_calls=0, new_PHY_calls=0, new_noise_draws=0, new_bootstrap=0,
        manual_visual_QA_claimed=False, outputs=outputs))
    print(json.dumps(dict(status='RENDERED', out=str(out), figure_count=3, files=len(outputs)+1), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--swin-frames', required=True)
    parser.add_argument('--p-frames', required=True)
    parser.add_argument('--bpg-frames', required=True)
    parser.add_argument('--var-frames', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
