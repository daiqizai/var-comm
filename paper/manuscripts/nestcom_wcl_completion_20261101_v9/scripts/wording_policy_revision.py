"""Apply the 2026-11-03 terminology, policy-index and cost presentation revision."""
def apply_wording_policy_revision(s, P):
    old = ('The method and frozen policy directory are common protocol context; neither depends on the source. '
           'The receiver obtains the actual endpoint from the received profile, not from a transmitter witness.')
    new = ('Under a preconfigured policy, both terminals know the method, MCS and profile directory. '
           'The 12-bit profile identifies the actual endpoint within that directory; it does not identify a new directory or signal an MCS change. '
           'A policy change requires both terminals to configure the corresponding directory before sending data frames. '
           'The receiver reads the endpoint from the received header and uses the configured MCS for body demapping and decoding.')
    assert s.count(old) == 1
    s = s.replace(old, new)
    old = 'The original 3378-profile directory and its added NULL entries remain the identified context of historical controls.'
    new = ('Earlier studies used the joint endpoint/MCS catalogues identified below: within those shared catalogues, a profile entry also contained MCS parameters. '
           'Their catalogue identity remains part of the protocol context. The current U directories instead fix one MCS each.')
    assert s.count(old) == 1
    s = s.replace(old, new)
    s = s.replace(r'\subsection{Candidate pools and calibration}',
                  (P/'tables/policy_experiment_index.tex').read_text()+'\n'+r'\subsection{Candidate pools and calibration}', 1)
    # Keep final settings and effective validations in S7; preserve operational history separately.
    for prefix in ['The previous affinity [38,39]', 'Three unsuccessful compilation attempts']:
        para = next(x for x in s.split('\n\n') if x.startswith(prefix))
        s = s.replace(para+'\n\n', '')
    s = s.replace('The timing sessions below retain their original registered ceilings, MCSs and numerical routes. The new calibration does not remeasure latency; these measurements are not attached as the timing of a different newly selected policy.',
                  'The timings below use the H10, H7, L10 and C1024 configurations indexed in S1.3. They measure complete implementations, including candidate generation and scanning. Current N1024 quality uses the separately frozen U policies. Implementation and execution records are archived in the repository reproduction log.')
    s = s.replace("Ada's measurements below document our adaptation only and are withheld from the main timing comparison for the implementation reasons given below.",
                  'The qualified compiled Ada implementation is reported separately in S7.4, with its corresponding quality in S7.5.')
    s = '\n'.join(line for line in s.splitlines() if not line.startswith('Ada (audit only) & 7653'))+'\n'
    s = s.replace('The six main methods each complete 96 measured RX calls successfully.',
                  'Each listed implementation completes 96 measured RX calls successfully.')
    s = s.replace('Before successful new quality evaluation, an additional fixed-original-seed arm is specified. It needs 30 extra tail renders and reuses two identical cases.',
                  'A preregistered fixed-original-seed arm isolates the effect of payload-derived generation seeds, using 30 additional tail renders and two exact reuses.')
    s = s.replace('The original serial quality route was selected for unified calibration before this diagnostic\'s outcome.',
                  'The original common500 quality route remains fixed.')
    a = s.index('The frozen Ada adaptation evaluates probabilities serially')
    b = s.index(r'\subsection{Prefill compatibility diagnostic}', a)
    s = (s[:a]+'These grid timings include shared scale-wise probability evaluation, independent termination and complete candidate scanning. Earlier serial-Ada and two-source instrumentation diagnostics are retained in '+
         r'\path{reproduction_logs/timing_execution_history_20261103.md}'+
         '; they are not used to adjust the 32-source measurements.\n\n'+s[b:])
    # Flush the historical full-width example floats before the processing section.
    s = s.replace(r'\section{Unified processing costs}', r'\clearpage'+'\n'+r'\section{Unified processing costs}', 1)
    return s
