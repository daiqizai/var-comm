"""Replace the final draft gate with actual timing values; no new measurements."""
from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parents[1]
path=P/'inputs/e5/timing_summary.csv'
c=json.loads((P/'inputs/e5/execution_closure.json').read_text())
assert c['actual_wait_returncode']==0 and c['owner_gone'] and c['child_gone']
assert c['summary_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
with path.open(newline='')as f:rows=list(csv.DictReader(f))
def ms(group,direction):
 a=[r for r in rows if r['group']==group and r['direction']==direction];assert len(a)==1
 return 1000*float(a[0]['equal_source_mean_seconds'])
def pair(group):return f"{ms(group,'TX'):.1f}/{ms(group,'RX'):.1f}"
grid='/'.join(f"{ms(g,'TX'):.1f}"for g in ['granularity_whole','granularity_quarter','granularity_eighth','Token_NeST_N1024_S10'])
text=(r'Table~\ref{tab:external} reports fresh processing costs on 32 sources, with two warmups and three measurements per source, batch one, one RTX PRO 6000 and two CPU cores. '
      f"At $N=1024$, 10 dB, NeST TX/RX means are {pair('Token_NeST_N1024_S10')} ms, versus {pair('WHOLE_N1024_S10')} ms for WHOLE and {pair('Ada_N1024_S10')} ms for Ada. "
      f"Under the matched ceiling, Whole/Quarter/Eighth/Token TX means are {grid} ms. "
      r'Timing includes fresh probability construction, independent termination, full candidate scanning, and actual link decoding; it excludes model loading, disk I/O and quality scoring. Supplement~S7 gives all 16 groups, median/p95 and successful-RX costs. These are software processing costs, not radio latency or optimized-hardware bounds.')
p=P/'sections/pending_results.tex';old=p.read_text()
marker=r'\textbf{Draft completion gate:}'
assert old.count(marker)==1
p.write_text(old.split(marker)[0]+text+'\n',encoding='utf8',newline='\n')
print(json.dumps(dict(status='ACTUAL_E5_INTEGRATED',NeST_N1024_S10_TX_RX_ms=pair('Token_NeST_N1024_S10'),matched_grid_TX_ms=grid)))
