"""Add compact navigation and a self-contained timing configuration explanation."""
import re

def apply_reading_flow_revision(s, P=None):
    marker='% Reading-flow navigation 20261103'
    if marker not in s:
        intro='Implementation recipe, complete results, paired statistics and source examples. Numerical tables use the same frozen CSV sources as the main manuscript.'
        nav=r'''
% Reading-flow navigation 20261103
\textbf{Contents}\par
\begin{tabular}{@{}p{0.47\linewidth}p{0.47\linewidth}@{}}
S1 Algorithms, proofs and implementation (p.~\pageref{sup:1}) & S5 Fixed-policy channel threshold (p.~\pageref{sup:5})\\
S2 Complete quality results (p.~\pageref{sup:2}) & S6 Qualitative examples and selection (p.~\pageref{sup:6})\\
S3 Matched controls and generation (p.~\pageref{sup:3}) & S7 Processing costs and paired quality (p.~\pageref{sup:7})\\
S4 Equal information budgets (p.~\pageref{sup:4}) & S8 Transfer to Kodak24 (p.~\pageref{sup:8})
\end{tabular}
'''
        assert s.count(intro)==1
        s=s.replace(intro,intro+'\n'+nav,1)
        sections=list(re.finditer(r'\\section\{[^}]+\}',s));assert len(sections)==8
        for i,m in reversed(list(enumerate(sections,1))):s=s[:m.end()]+r'\label{sup:'+str(i)+'}'+s[m.end():]
    old='The timings below use the H10, H7, L10 and C1024 configurations indexed in S1.3. They measure complete implementations, including candidate generation and scanning. Current N1024 quality uses the separately frozen U policies. Implementation and execution records are archived in the repository reproduction log.'
    new=r'''At $N=1024$ and 10 dB, the token-system timing runs use 16-QAM rate 2/3 and target ceilings $(9,0)$, $(8,126)$ and 256 tokens for WHOLE, NeST and Ada, respectively. These MCSs and ceilings match the current main quality experiment; the profile directories are the earlier versions indexed in S1.3. Compiled Ada's latency is paired with its own 32-image quality evaluation in S7.5. Additional timings cover 7 dB, $N=512$ and matched granularity. The H10/H7/L10/C1024 identifiers in S1.3 link these measurements to exact directory versions. All timings measure complete implementations, including candidate generation and scanning; execution records remain in the repository reproduction log.'''
    if old in s:s=s.replace(old,new,1)
    else:assert new in s
    terms={
        'Integer probabilities and independently terminated streams': 'Integer CDFs and independently terminated prefixes',
        'Actual source lengths and endpoint saturation': 'Actual source-code lengths and endpoint saturation',
        'Traverse source tokens once, sharing probability evaluation and incremental coding. At each allowed endpoint copy its independently terminated bits without changing encoder state.': 'Traverse source tokens once, sharing scale-wise probability evaluation and incremental coding. At each allowed endpoint obtain an independently terminated prefix without changing encoder state.',
    }
    for old,new in terms.items():
        s=s.replace(old,new)
    return s
