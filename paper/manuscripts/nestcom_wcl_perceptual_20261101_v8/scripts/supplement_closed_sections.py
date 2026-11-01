"""Insert only hash-bound, actually closed E2/E6 evidence into the supplement."""
import csv, json, hashlib


def enrich(text, root, table):
    def rows(stage, name):
        with (root/'inputs'/stage/name).open(newline='', encoding='utf8') as f:
            return list(csv.DictReader(f))

    for stage in ['e2', 'e6']:
        path = root/'inputs'/stage
        closure = json.loads((path/'execution_closure.json').read_text())
        assert closure['actual_wait_returncode'] == 0 and closure['owner_gone'] and closure['child_gone']
        assert not closure['failure_present']
        assert hashlib.sha256((path/'summary.csv').read_bytes()).hexdigest() == closure['summary_sha256']
    labels = {'Token_NeST': 'NeST', 'WHOLE': 'WHOLE', 'Ada': 'Ada'}
    metrics = {'psnr_db': 'PSNR', 'lpips_alex': 'LPIPS', 'dreamsim': 'DreamSim',
               'clip_image_cosine': 'CLIP-I', 'dists': 'DISTS'}
    e2 = rows('e2', 'summary.csv'); delta = rows('e2', 'paired_deltas.csv')
    assert len(e2) == len(delta) == 90
    e2_text = r'''All 15\,000 logical outputs were accepted by the actual source receiver; no NULL or gray output occurs in this error-free diagnostic. New reconstruction requires 195 NeST/WHOLE receiver calls and 8733 Ada calls; 4466 NeST/WHOLE and 225 Ada input identities are reused. There are 8882 newly scored native pairs; 6118 logical positions reuse exact completed metric pairs. These counts are distinct from the number of information-transfer outcomes.

At every budget all five direct NeST--WHOLE intervals favor NeST. NeST--Ada PSNR remains positive throughout. DISTS favors NeST at 620/1212/1302/1940 bits and Ada at 2248/2577 bits. DreamSim favors NeST at 620/1212/1302 bits, crosses zero at 1940 bits, and favors Ada at 2248/2577 bits. LPIPS favors NeST through 2248 bits and crosses zero at 2577 bits. CLIP-I favors Ada at 620/1940/2577 bits, with intervals crossing zero at the other budgets. These mixed results remain part of the complete-system comparison. Increasing the budget from 1212 to 1302 bits raises mean source bits from 915.858 to 918.954 for WHOLE and from 1150.046 to 1239.950 for NeST. Thus NeST uses 89.904 of the extra 90 bits on average, while WHOLE uses 3.096; this is a source mean, not a guarantee for every image.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/supp_figure_s1_information_budget.pdf}
\caption{Error-free information-budget comparison on common500. Total bits include all registered information overhead. Numeric budget positions retain their unequal spacing. Points and bars are equal-source means and pointwise 95\% source-bootstrap intervals. Ada's three generation repeats are averaged within source; they are not channel repeats. Lower distances are favorable.}\label{fig:information-budget}
\end{figure}

\subsection{Complete means and direct paired differences}
'''
    body = [[v['B_bits'], labels[v['method']], metrics[v['metric']],
             f"{float(v['mean']):.5f} [{float(v['ci95_low']):.5f}, {float(v['ci95_high']):.5f}]"] for v in e2]
    e2_text += table(['Bits', 'Method', 'Metric', r'Mean [95\% CI]'], body, 'rlll')
    body = [[v['B_bits'], labels[v['candidate']]+'--'+labels[v['reference']], metrics[v['metric']],
             f"{float(v['mean_delta']):+.5f} [{float(v['ci95_low']):.5f}, {float(v['ci95_high']):.5f}]"] for v in delta]
    e2_text += table(['Bits', 'Comparison', 'Metric', r'Difference [95\% CI]'], body, 'rlll')
    cap = rows('e2', 'representation_and_capacity.csv'); assert len(cap) == 18
    e2_text += r'\subsection{Actual source lengths and endpoint saturation}'+'\n'
    body = [[v['B_bits'], labels[v['method']], f"{float(v['mean_actual_stream_bits']):.2f}",
             f"{float(v['mean_padding_bits']):.2f}", v['full_representation_sources'],
             v['no_feasible_prefix_sources']] for v in cap]
    e2_text += table(['Bits', 'Method', 'Source bits', 'Padding bits', 'Full / 500', 'NULL / 500'], body, 'rlrrrr')
    e2_text += ('Full means all 10 NeST/WHOLE scales or all 256 Ada tokens. These counts are computed '
                'once per source, not once per Ada repeat. Padding after a complete representation is not '
                'failure to fit a prefix. Source bits exclude the separately stated information overhead.\n')
    old = r'\textbf{Draft: the Ada reconstruction stage and resulting complete five-metric statistics remain pending. No quality result is filled from partial outputs.}'
    assert text.count(old) == 1
    text = text.replace(old, e2_text)

    e6 = rows('e6', 'summary.csv'); assert len(e6) == 15
    e6_text = r'''All 30\,000 real receptions and both distance metrics have completed. There are 17\,497 accepted outcomes and 12\,503 body/profile-parse rejections, with no header rejection, unexpected accepted source input, source-decoding rejection or NULL. No additional image generation is needed: accepted prefixes and frozen generation identities match existing outputs. Scoring required 88 new native gray-reference pairs for each distance; all other pairs were reused after identity verification.

All methods reject all 2000 receptions at 8 dB. At 8.5 dB their rejection rates are 93.05--94.00\%, at 9 dB they are 13.50--15.75\%, and at 9.5 dB they are 0.10--0.20\%. No rejection is observed at 10 dB. Source-cluster intervals retain all 20 noise repetitions within each sampled source. This finite scan describes the tested fixed policies, not low-error-rate operation or general channel robustness.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/supp_figure_s2_threshold.pdf}
\caption{Frozen 10-dB policies evaluated at five actual SNRs. Points and bars retain all receptions and show pointwise 95\% source-bootstrap intervals. Dotted quality curves describe successful receptions only and use a changing event denominator; no conditional point exists at 8 dB.}\label{fig:threshold}
\end{figure}
'''
    body = []
    for v in e6:
        for key, label in [('frame_rejection_rate', 'Rejection rate'), ('dists', 'DISTS'), ('dreamsim', 'DreamSim')]:
            body.append([labels[v['method']], v['snr_db'], label,
                         f"{float(v[key]):.5f} [{float(v[key+'_ci95_low']):.5f}, {float(v[key+'_ci95_high']):.5f}]"])
    e6_text += table(['Method', 'SNR', 'Metric', r'Overall mean [95\% CI]'], body, 'lrll')
    body = [[labels[v['method']], v['snr_db'], v['total_success'],
             f"{float(v['dists_successful_only']):.5f}" if v['dists_successful_only'] else '--',
             f"{float(v['dreamsim_successful_only']):.5f}" if v['dreamsim_successful_only'] else '--'] for v in e6]
    e6_text += table(['Method', 'SNR', 'Success / 2000', 'Cond. DISTS', 'Cond. DreamSim'], body, 'lrrrr')
    old = r'\textbf{Draft: the 30\,000 physical events have exited successfully, but the complete distance/conditional-quality summary awaits the final missing gray-image scores.} Successful payload identities are authenticated before image reuse; no unexpected accepted source input occurred. These points do not establish low-error-rate operation or general channel robustness.'
    assert text.count(old) == 1
    text = text.replace(old, e6_text)
    old = r'\textbf{Working draft.} This version contains complete E1/E3/E4 data and explicit gates for pending E2/E5/E6 deliveries. It is not a completed publication artifact.'
    assert text.count(old) == 1
    text = text.replace(old, r'\textbf{Working draft.} E1--E4 and E6 contain complete results. Unified timing and final document acceptance remain pending.')
    old = r'\textbf{Draft: the timing queue waits for this round\'s reconstruction/scoring jobs to finish before taking exclusive use of the designated GPU. All 16 groups and 32 TX/RX stages remain pending.}'
    # The exact original uses a literal apostrophe, not a TeX escape.
    old = old.replace("\\'", "'")
    assert text.count(old) == 1
    text = text.replace(old, r'\textbf{Draft: unified timing remains incomplete. No historical timings or partial measurements are substituted for the required 16 groups and 32 TX/RX stages.}')
    return text
