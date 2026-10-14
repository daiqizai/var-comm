"""Validate this PLAN's arithmetic and limits. Does not load models or run PHY."""
from pathlib import Path
from fractions import Fraction
import json

p=Path(__file__).with_name('scope.json')
x=json.loads(p.read_text(encoding='utf-8'))
assert x['status']=='PLAN_ONLY_NOT_IMPLEMENTED_OR_EXECUTED'
s=x['scenario']; assert s['header_uses']+s['body_uses']==s['N_complex_uses']==1024
assert sum(s['mixture'].values())==1
assert x['new_training_updates']==0
ends=x['source']['endpoint_grid']; assert len(ends)==24
assert len({(e['m'],e['K']) for e in ends})==24
sz=x['source']['scale_sizes']; totals=[sum(t*t for t in sz[:j]) for j in range(11)]
assert [totals[j] for j in (5,6,7,8,9,10)]==[55,91,155,255,424,680]
assert all(e['tokens']==totals[e['m']]+e['K'] for e in ends)
assert sum(sum(e['tokens']>totals[j] for e in ends) for j in (4,5,6,7))==68
p2=x['LE2_caps']; trials=p2['new_packet_snr_cells_initial']*p2['trials_per_initial_cell']+p2['max_topped_up_cells']*(p2['final_trials_per_topped_up_cell']-p2['trials_per_initial_cell'])
assert trials==p2['max_new_packet_trials']==21504
p3=x['LE3_caps']; assert sum(p3['candidate_slots'].values())==10
assert p3['pilot_sources']*p3['true_snr_count']*p3['noise_count']*p3['max_unique_physical_policies']==3000==p3['max_image_physical_frames']
p4=x['LE4_conditional']; assert p4['sources']*p4['methods']*len(p4['true_snrs_db'])*p4['noise_count']==4500==p4['physical_frames']
assert len(x['candidate_grid']['base_symbol_grid_coarse'])==17
for r in x['candidate_grid']['nominal_rates']:
    assert Fraction(0)<Fraction(r)<=Fraction(5,6)
print('PASS: plan arithmetic, 24 endpoints, 68 segment edges, and finite stage limits.')
print('This check is not code qualification, a model test, or an experiment result.')
