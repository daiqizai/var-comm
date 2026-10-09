"""Format the completed A3/A4 CSVs as booktabs fragments; no new statistics."""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

INPUTS = {
    'timing': ('results/paper_supplement_20261008/a3_timing/final_v1/timing_single_output.csv',
               'ff44c9352b1b8dbc68c62e24b1796ea2133136178c5f8c542f0776001e4bb435'),
    'storage': ('results/paper_supplement_20261008/a3_timing/final_v1/model_storage.csv',
                '006f903d0f82ab629d46120cf70918257c226c1f50d2a07f88a808d6c6ff291d'),
    'configs': ('results/paper_supplement_20261008/a4_resources/v2/resource_summary.csv',
                '3dd6b0de3661db6782ca78f54ea0bc2b75c60d7f45224ae64bc5ac1b6e22c08b'),
}
METHODS = [
    ('RAW64_PARTIAL','Partial-scale digital + VAR'),
    ('RAW64_WHOLE','Full-scale digital + VAR'),
    ('P1024','Latent continuous JSCC'),
    ('SwinJSCC80k','SwinJSCC-80k (adapted)'),
    ('BPG_LDPC_native256','BPG + LDPC (native 256)'),
]
SNRS = [7,13,19]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, places=2):
    d=Decimal(str(value)); require(d.is_finite(),'Nonfinite table value')
    return format(d.quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_UP),f'.{places}f')


def pair(row, prefix):
    return number(row[prefix+'_mean_ms'])+' / '+number(row[prefix+'_p95_ms'])


def row(cells):
    return ' & '.join(cells)+r' \\'+'\n'


def main(args):
    root=Path(args.root).resolve(); out=Path(args.output).resolve() if args.output else root/'paper/tables/paper_supplement_20261009'
    tables={};bindings={}
    for key,(relative,expected) in INPUTS.items():
        p=root/relative; require(sha(p)==expected,'Changed completed CSV: '+relative)
        tables[key]=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')));bindings[relative]=expected
    timing={(r['method'],int(r['snr_db'])):r for r in tables['timing']}
    storage={r['method']:r for r in tables['storage']}
    require(len(timing)==len(tables['timing'])==15 and set(timing)=={(m,s) for m,_ in METHODS for s in SNRS},
            'Five methods x three SNRs required')
    require(len(storage)==len(tables['storage'])==5 and set(storage)=={m for m,_ in METHODS},'Five storage identities required')
    for (method,s),r in timing.items():
        require(int(r['source_count'])==16 and int(r['measured_attempts'])==48 and int(r['warmups_excluded'])==16,
                'Fixed16 repeated-measurement scope changed')
        n=int(r['actual_link_attempts']);unfit=int(r['source_unfit_attempts'])
        require(n+unfit==48 and all(int(r[p+'_available_samples'])==n for p in ['tx','rx','software_e2e_excluding_channel']),
                'Available link-window counts differ')
        if method=='BPG_LDPC_native256':
            require(n=={7:3,13:12,19:30}[s] and int(r['source_encoding_attempt_available_samples'])==unfit
                    and int(r['failed_source_attempt_elapsed_available_samples'])==unfit,'Native BPG conditional scope changed')
        else:
            require(n==48 and unfit==0,'Unexpected omitted neural timing samples')
    configs={(r['method'],int(r['snr_db'])):r for r in tables['configs']}
    require(len(configs)==len(tables['configs'])==24,'Full A4 resource table required')
    for family in ['RAW_WHOLE','RAW_PARTIAL']:
        for s in [1,4,7,10,13,19]:
            r=configs[family,s]
            require(int(r['source_count'])==500 and int(r['noise_count'])==3,'Original holdout scope changed')
            require(sum(int(r[k]) for k in ['header_complex_uses','body_transmitted_complex_uses','idle_complex_uses'])==1024,
                    'Digital budget must be exactly1024')
            require(r['ldpc_effective_rate_fraction']==r['ldpc_information_k']+'/'+r['ldpc_transmitted_n'],
                    'Actual code fraction differs from integer lengths')

    text=r'''% Generated from hash-pinned completed CSVs. Requires \usepackage{booktabs}.
% Milliseconds, directly formatted from original means/P95; no new estimation.
\begin{table*}[t]
\centering
\caption{Single-output software timing in the same measurement environment.
Each cell gives mean / empirical P95 in milliseconds. There are 16 fixed sources,
one excluded warmup and three measured repetitions per source and SNR.
End-to-end (E2E) uses the measured outer window with software channel time
subtracted; it is not TX+RX and is not air-interface latency.
Model loading, file I/O and quality scoring are excluded; endpoint device copies
are included. The repeated measurements do not constitute independent noise trials.}
\label{tab:single-output-timing}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrccc@{}}
\toprule
Method & \shortstack{SNR\\(dB)} & \shortstack{Measured links\\/ attempts} & \shortstack{TX (ms)\\mean / P95} & \shortstack{RX (ms)\\mean / P95} & \shortstack{Software E2E (ms)\\mean / P95} \\
\midrule
'''
    for mi,(method,label) in enumerate(METHODS):
        for s in SNRS:
            r=timing[method,s];n=str(r['actual_link_attempts'])+'/48'
            printed_label=label if s==SNRS[0] else ''
            if method=='BPG_LDPC_native256':
                n=r'\textbf{'+n+'}'
                if printed_label:printed_label+=r'$^{\dagger}$'
            snr=str(s)+(r'$^{\ddagger}$' if method=='SwinJSCC80k' and s==19 else '')
            text+=row([printed_label,snr,n,pair(r,'tx'),pair(r,'rx'),pair(r,'software_e2e_excluding_channel')])
        if mi<len(METHODS)-1:text+='\\addlinespace[2pt]\n'
    text+=r'''\bottomrule
\end{tabular*}
\par\vspace{2pt}
\begin{minipage}{\textwidth}\footnotesize
$^{\dagger}$\textbf{BPG link timings are conditional on an actual transmitted
frame: only 3, 12 and 30 of 48 attempts at 7, 13 and 19 dB}, corresponding to
1, 4 and 10 distinct sources. They are not full-population deployment averages.
Failed source-coding costs are reported separately below; unavailable RX/E2E
values are not replaced with zero. P95 is descriptive for these small samples.
$^{\ddagger}$19 dB is outside Swin's training and calibration range.
HiFi-DiffCom timing from a different environment is not included.
\end{minipage}
\end{table*}

\begin{table}[t]
\centering
\caption{Native256 BPG source-capacity failures. These attempts run real source
coding but produce no transmitted frame, RX or link E2E measurement. Costs are
mean / empirical P95 in milliseconds. The failed-attempt outer window and
source-codec component are alternatives, not additive costs.}
\label{tab:bpg-source-unfit-cost}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}rrcc@{}}
\toprule
\shortstack{SNR\\(dB)} & \shortstack{Unfit\\/ attempts} & \shortstack{Source coding\\mean / P95} & \shortstack{Attempt elapsed\\mean / P95} \\
\midrule
'''
    for s in SNRS:
        r=timing['BPG_LDPC_native256',s]
        text+=row([str(s),str(r['source_unfit_attempts'])+'/48',pair(r,'source_encoding_attempt'),pair(r,'failed_source_attempt_elapsed')])
    text+=r'''\bottomrule
\end{tabular}
\end{table}

\begin{table*}[t]
\centering
\caption{Loaded model and stored-file footprint for the measured methods.
Parameters count unique loaded objects. Parameter tensors, checkpoint files and
the classical codec library are different quantities; checkpoint files may
include metadata. These are not minimal deployable sizes or energy measurements.
The BPG row refers to native256 BPG, not the later adaptive-downsampling baseline.}
\label{tab:single-output-storage}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
Method & \shortstack{Parameters\\(million)} & \shortstack{Parameter tensors\\(MiB)} & \shortstack{Checkpoint files\\(GiB)} & \shortstack{Codec library\\(MiB)} \\
\midrule
'''
    for method,label in METHODS:
        r=storage[method]
        require(r['minimal_deployment_size_claimed']=='False','Storage scope must remain loaded, not minimal')
        params=number(Decimal(r['unique_loaded_parameters'])/Decimal(1000000),3)
        tensor=number(Decimal(r['unique_parameter_bytes'])/Decimal(2**20),2)
        checkpoint=number(Decimal(r['total_checkpoint_file_bytes'])/Decimal(2**30),3)
        codec=number(Decimal(r['codec_library_bytes'])/Decimal(2**20),2) if r['codec_library_bytes'] else '--'
        text+=row([label,params,tensor,checkpoint,codec])
    text+=r'''\bottomrule
\end{tabular*}
\end{table*}
'''

    configtext=r'''% Generated from the unchanged A4 CSV; original500 x3-noise scope.
% Requires \usepackage{booktabs}. Actual k/n is not the nominal rate label.
\begin{table*}[t]
\centering
\caption{Frozen digital configurations at N=1024. $Q_m=\log_2 M$ is the number
of coded bits per modulation symbol. The actual LDPC rate is $k/n$, where $k$
includes the 16-bit body CRC. H/B/P denotes header/body/known-padding complex
uses and sums to1024. Full-scale and partial-scale policies have the same legal
modulation and coding permissions. A selected padded configuration does not
imply that the full-scale control lacked a full-budget protection option.}
\label{tab:selected-digital-configs}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rlrrrlcl@{}}
\toprule
SNR (dB) & Scheme & $m$ & $K$ & $Q_m$ & Actual $k/n$ & H/B/P & Allocation class \\
\midrule
'''
    classes={'full_budget':'Full budget','nominal':'Nominal','full_budget|nominal':'Both'}
    for si,s in enumerate([1,4,7,10,13,19]):
        for family,label in [('RAW_WHOLE','Full-scale'),('RAW_PARTIAL','Partial-scale')]:
            r=configs[family,s]
            q={'QPSK':'2','16QAM':'4','64QAM':'6'}[r['body_modulation']]
            allocation=classes[r['allocation_modes']]
            budget='/'.join(r[k] for k in ['header_complex_uses','body_transmitted_complex_uses','idle_complex_uses'])
            configtext+=row([str(s),label,r['m'],r['K'],q,r['ldpc_effective_rate_fraction'],budget,allocation])
        if si<5:configtext+='\\addlinespace[2pt]\n'
    configtext+=r'''\bottomrule
\end{tabular*}
\par\vspace{2pt}
\begin{minipage}{\textwidth}\footnotesize
``Both'' means that nominal and full-budget construction labels identify the
same transmitted action, not two frames. Known padding is energized QPSK and
carries no image information; it is not extra FEC or silent energy saving.
Partial-scale selection permits $K=0$ and retains the full-scale option: the
7 dB configuration is identical for both schemes.
\end{minipage}
\end{table*}
'''
    out.mkdir(parents=True,exist_ok=True)
    (out/'timing_single_output.tex').write_text(text,encoding='utf-8')
    (out/'selected_digital_configs.tex').write_text(configtext,encoding='utf-8')
    readme='''# Paper supplement tables (2026-10-09)

Generated directly from the completed, hash-pinned CSVs listed below. No new
training, inference, source coding, channel simulation, metric computation,
bootstrap, significance test or PDF compilation was performed. Decimal values
are formatted with ROUND_HALF_UP; original CSV precision is unchanged. Timing
mean/P95 uses two decimal places (ms). No value is clamped: a sample mean can
exceed an empirical P95 when a small upper tail is sufficiently large.

`timing_single_output.tex` contains three booktabs table fragments: five-method
TX/RX/software-E2E timing at7/13/19 dB, a separate BPG source-unfit cost table,
and loaded/stored footprint. BPG link sample counts3/12/30 (out of48) are bold
and explicitly conditional on transmitting. Each SNR has48 measured attempts
plus16 excluded warmups (three measured repetitions per source); repetitions are not
independent source/noise draws. SOURCE_UNFIT has no RX/E2E. Its coding cost is
shown separately, not replaced with zero. No HiFi timing from another environment
is mixed into the table. The BPG timing is native256, not adaptive downsampling.

`selected_digital_configs.tex` contains all12 raw digital configurations from
the original500-source/three-noise result: actual k/n, m/K, modulation bits per
symbol, header/body/padding and nominal/full-budget construction class. Printed
names are Full-scale and Partial-scale; internal file IDs remain in the script.

Use `\\usepackage{booktabs}` and `\\input{paper/tables/paper_supplement_20261009/timing_single_output.tex}`
or the configuration fragment. Main timing/storage/configuration tables use
`table*` with `tabular*{\\textwidth}` for a conventional double-column paper.
The four-column BPG failure-cost table is a normal single-column `table`.
No resizebox, landscape rotation, makecell, multirow or nonstandard font is needed.
These are LaTeX fragments, not standalone documents. They have not been compiled,
as requested; check final placement within the manuscript's class and margins.

Reproduce from the repository root:

    python experiments/paper_supplement_20261008/export_paper_tables.py --root .

Optional `--output NEW_DIRECTORY` selects a separate output location. Re-running
only overwrites these generated table fragments and metadata, never source CSVs.
Existing publication filters are unchanged.

## Input identities

'''
    for relative,digest in bindings.items():readme+=f'- `{relative}`: SHA256 `{digest}`.\n'
    (out/'README.md').write_text(readme,encoding='utf-8')
    require(all(sha(root/p)==h for p,h in bindings.items()),'Input changed during formatting')
    write={p.name:sha(p) for p in [out/'timing_single_output.tex',out/'selected_digital_configs.tex',out/'README.md']}
    (out/'export_manifest.json').write_text(json.dumps(dict(status='CSV_TO_LATEX_COMPLETE',inputs=bindings,outputs=write,
        timing_rows=15,digital_configuration_rows=12,storage_rows=5,new_measurements=0,new_statistical_tests=0,
        pdf_compiled=False,script_sha256=sha(__file__)),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',output=str(out),timing_rows=15,config_rows=12,pdf_compiled=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--output')
    main(parser.parse_args())
