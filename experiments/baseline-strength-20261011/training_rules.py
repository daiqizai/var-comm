"""Predeclared continuation schedules and two-window empirical stability."""
P_SNRS=[1,4,7,13,19]
S_SNRS=[1,4,7,10,13,19]

def learning_rate(branch,step,start,initial_lr):
    if branch=='P':
        if not 40000<step<=100000:raise ValueError('P step outside registered continuation')
        return 2e-4 if step<=60000 else 1e-4 if step<=70000 else 3e-5
    u=step-start
    if not 0<u<=120000 or step>240000:raise ValueError('Swin step outside hard limit')
    return min(initial_lr,1e-4 if u<=40000 else 3e-5 if u<=60000 else 1e-5)

def stability(history,step,width):
    table={r['step']:r for r in history};checks=[]
    for end in [step-width,step]:
        if end not in table or end-width not in table:raise ValueError('Missing full calibration endpoint')
        a,b=table[end-width],table[end]
        for key in ['overall',*sorted(a['cells'])]:
            before=a['objective'] if key=='overall' else a['cells'][key]
            after=b['objective'] if key=='overall' else b['cells'][key]
            rel=(after-before)/max(before,1e-12)
            checks.append(dict(start=end-width,end=end,cell=key,before=before,after=after,signed_relative_change=rel,absolute_relative_change=abs(rel),passes=abs(rel)<.002))
    return dict(stable=all(x['passes'] for x in checks),checks=checks,
                improving=any(x['signed_relative_change']<=-.002 for x in checks))
