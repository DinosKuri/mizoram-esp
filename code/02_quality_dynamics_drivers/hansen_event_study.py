"""Event study: RSEI around Hansen GFC forest-loss events (100 m cells).
Events: loss fraction >= 0.5 in one year 2004-2020, tree cover 2000 >= 50%.
Control: undisturbed forest (no loss, tree cover 2000 >= 70%).
Season t = loss year (dry season Nov t-1 to Feb t). Anomaly = (RSEI - control mean of that season),
minus the cell's own mean anomaly over seasons t-3..t-1. Observed (not gap-filled) RSEI only."""
import numpy as np, pandas as pd, json
v = np.load('validation100.npz'); s = np.load('rsei_stack100_obs.npz')
yrs = [int(k) for k in s.files]; R = np.stack([s[k] for k in s.files]).astype('float32')
lf, ly, tc = v['loss_frac'], v['loss_year'], v['tc2000']
ctrl = (lf == 0) & (tc >= 70)
cm = np.array([np.nanmean(R[i][ctrl]) for i in range(len(yrs))])
A = R - cm[:, None, None]
ev = (lf >= 0.5) & (tc >= 50) & (ly >= 4) & (ly <= 20)
iy, ix = np.where(ev); L = 2000 + ly[iy, ix].astype(int)
yi = {y: i for i, y in enumerate(yrs)}
def get(k):
    out = np.full(len(L), np.nan, 'float32')
    for y in np.unique(L):
        if y + k in yi:
            m = L == y; out[m] = A[yi[y + k], iy[m], ix[m]]
    return out
pre = np.nanmean(np.stack([get(k) for k in (-3, -2, -1)]), 0)
rows = []
for k in range(-3, 6):
    d = get(k) - pre; d = d[np.isfinite(d)]
    rows.append(dict(rel_season=k, mean=d.mean(), sd=d.std(), n=d.size))
pd.DataFrame(rows).to_csv('hansen_event_study.csv', index=False)
print(pd.DataFrame(rows))
