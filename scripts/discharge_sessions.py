"""Classify discharge sessions (ac_out_pwr > 5 kW, gaps <= SESSION_GAP_MIN bridged) as
major / minor by I^2t: sum of ac_out_current^2 * 60 s over the session, in MA^2s.
>= 2 MA^2s is major, otherwise minor. Prints counts per unit.

Usage: uv run python scripts/discharge_sessions.py
"""
import sys; sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
from analyze import load_unit_metrics, SESSION_GAP_MIN
u=load_unit_metrics()
print(u[['ac_out_pwr','ac_out_current']].describe().round(1).to_string())
TH, MAJOR = 5.0, 2.0  # kW ; MA^2s
out=[]
for ep,g in u.groupby('endpoint_id'):
    g=g.sort_values('ts_min').set_index('ts_min')
    p=g['ac_out_pwr']; active=p>TH
    run=(active!=active.shift()).cumsum(); rl=active.groupby(run).transform('size')
    br=active|(~active&(rl<=SESSION_GAP_MIN))
    sid=(br!=br.shift(fill_value=False)).cumsum()
    i2t=(g['ac_out_current'].fillna(0)**2*60)/1e6   # A^2 * 60 s per 1-min sample -> MA^2s
    d=pd.DataFrame({'a':br,'sid':sid,'i2t':i2t}).groupby('sid').agg(a=('a','first'),i2t=('i2t','sum'))
    d=d[d.a].assign(unit=ep); out.append(d)
d=pd.concat(out); d['class']=np.where(d.i2t>=MAJOR,'major','minor')
print(d.groupby(['unit','class']).size().unstack(fill_value=0).assign(total=lambda x:x.sum(axis=1)).to_string())
print(d['class'].value_counts().to_string())
print(d.i2t.describe(percentiles=[.1,.25,.5,.75,.9]).round(2).to_string())
