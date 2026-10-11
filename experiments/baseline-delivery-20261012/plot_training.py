"""Plot completed baseline training records; no model loading or resampling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(fig, out, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / (name + '.' + ext), dpi=600, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--p-history', type=Path, required=True)
    parser.add_argument('--swin-history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5,
                         'legend.fontsize': 7.5, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': .25, 'lines.linewidth': 1.25})
    records, data, bindings = [], [], {}
    for branch, dirname, path, start, end, color in (
        ('P1024', 'P_continue', args.p_history, 40000, 100000, '#0072B2'),
        ('SwinJSCC (adapted)', 'Swin_1to19', args.swin_history, 80000, 200000, '#D55E00'),
    ):
        folder = args.results / dirname
        for p in [path, folder/'completion.json', folder/'calibration_history.json', folder/'selected_checkpoint.json']:
            bindings[str(p)] = sha(p)
        assert json.loads((folder/'completion.json').read_text())['status'] == 'TRAINING_AND_CALIBRATION_COMPLETE'
        selected = json.loads((folder/'selected_checkpoint.json').read_text())
        cal = json.loads((folder/'calibration_history.json').read_text())
        assert selected['step'] == min(cal, key=lambda x: (x['objective'], x['step']))['step']
        rows = list(csv.DictReader(path.open()))
        steps = [int(r['global_step']) for r in rows]
        assert steps == list(range(start+1, end+1)), 'Missing or duplicate training updates'
        loss = np.array([float(r['train_loss']) for r in rows])
        assert np.isfinite(loss).all()
        blocks = []
        for i in range(0, len(rows), 500):
            block = rows[i:i+500]
            value = float(np.mean(loss[i:i+500], dtype=np.float64))
            blocks.append((int(block[-1]['global_step']), value))
            data.append(dict(branch=branch, series='training_loss_500_update_block_mean',
                             global_step=blocks[-1][0], value=value, count=len(block)))
        for row in cal:
            data.append(dict(branch=branch, series='calibration_objective', global_step=row['step'], value=row['objective'], count=row['rows']))
            for cell, value in row['cells'].items():
                data.append(dict(branch=branch, series='calibration_'+cell, global_step=row['step'], value=value, count=row['source_count']*row['noise_count']))
        records.append((branch, rows, cal, selected, blocks, color))

    fig, axes = plt.subplots(2, 2, figsize=(7, 4.9), layout='constrained')
    for j, (branch, rows, cal, selected, blocks, color) in enumerate(records):
        x, y = np.array(blocks).T
        axes[0, j].plot(x/1000, y, color=color)
        axes[0, j].set(title=branch, ylabel='Training objective', xlabel='Total updates (thousands)')
        axes[1, j].plot([r['step']/1000 for r in cal], [r['objective'] for r in cal], color=color, marker='o', markersize=3)
        axes[1, j].scatter([selected['step']/1000], [selected['objective']], color='black', marker='*', s=65, zorder=5, label='Selected checkpoint')
        axes[1, j].set(xlabel='Total updates (thousands)', ylabel='Full calibration objective')
        axes[1, j].legend(loc='best')
    save(fig, out, 'training_calibration_curves')

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), layout='constrained')
    colors = ['#000000', '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
    markers = ['o', 's', '^', 'v', 'D', 'x']
    for ax, (branch, rows, cal, selected, blocks, color) in zip(axes, records):
        cells = sorted((k for k in cal[0]['cells'] if k.startswith('N1024_')), key=lambda k: int(k.split('snr')[1]))
        for i, key in enumerate(cells):
            i = [1, 4, 7, 10, 13, 19].index(int(key.split('snr')[1]))
            ax.plot([r['step']/1000 for r in cal], [r['cells'][key] for r in cal], color=colors[i], marker=markers[i], markersize=2.5, linestyle=['-', '--', '-.', ':', '-', '--'][i], label=key.split('snr')[1]+' dB')
        ax.set(title=branch + ', N=1024', xlabel='Total updates (thousands)', ylabel='Calibration cell objective')
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + .28*(high-low))
        ax.legend(ncol=3, fontsize=7, loc='upper right', framealpha=1)
    save(fig, out, 'calibration_by_snr')

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), layout='constrained')
    for ax, (branch, rows, cal, selected, blocks, color) in zip(axes, records):
        ax.step([int(r['global_step'])/1000 for r in rows], [float(r['lr']) for r in rows], color=color, where='post')
        ax.set(title=branch, xlabel='Total updates (thousands)', ylabel='Learning rate')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    save(fig, out, 'learning_rate_schedule')
    with (out/'plot_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (out/'captions.tex').write_text(
        r'\paragraph{Training and calibration.} Training objectives are shown as non-overlapping means of 500 consecutive updates, including every recorded update. Calibration points are the complete frozen calibration evaluations. P1024 uses RGB MSE plus 0.1 LPIPS-Alex plus 0.01 normalized latent MSE; SwinJSCC uses RGB MSE. The objectives are not directly comparable across branches. P1024 satisfies the registered empirical plateau criterion at 100,000 total updates. SwinJSCC reaches 200,000 total updates (120,000 additional updates) while still improving; convergence is not claimed.'+'\n\n'+
        r'\paragraph{Calibration by SNR.} N1024 cell objectives versus total updates. P1024 selection uses five original SNR values, excluding 10 dB. SwinJSCC selection equally weights both original budgets and all six SNR values; only its N1024 cells are displayed here. All calibration failures remain in the denominators. No holdout score enters checkpoint selection.'+'\n\n'+
        r'\paragraph{Training settings.} P1024 resumes the original full 40,000-update state, including optimizer and random states. SwinJSCC is initialized from its 80,000-update weights with a fresh optimizer, sampling order, and random states, as authorized; its additional training covers integer SNRs from 1 to 19 dB. This changes both training coverage and optimization history. The original 80,000-update SwinJSCC result at 19 dB remains outside its training and calibration range. The new selected checkpoint covers 19 dB.'+'\n', encoding='utf-8')
    (out/'completion.json').write_text(json.dumps(dict(status='TRAINING_FIGURES_GENERATED', input_sha256=bindings,
        training_updates_plotted={'P': 60000, 'Swin': 120000}, bootstrap_calls=0, inference_calls=0,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'}), indent=2)+'\n')


if __name__ == '__main__':
    main()
