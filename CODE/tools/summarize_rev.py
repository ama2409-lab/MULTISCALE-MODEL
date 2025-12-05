import pandas as pd
import numpy as np
fn = r"FINAL SET/sweep_rev_results.csv"
df = pd.read_csv(fn)
# Convert L to numeric
df['L_m'] = pd.to_numeric(df['L_m'])
cols = ['dir_D_eff','neu_D_eff','porosity','tau_dir','tau_neu','cond_n_positive']
print('\nGrouped statistics by L (mean ± std, count):\n')
for L, g in df.groupby('L_m'):
    out = {c: f"{g[c].mean():.3e} ± {g[c].std():.3e} (n={len(g)})" for c in cols}
    print(f"L={L:.6g} m: ")
    for k,v in out.items():
        print(f"  {k}: {v}")
    print()
# Also show correlation between porosity and dir_D_eff
print('Correlation between porosity and dir_D_eff:')
print(df['porosity'].corr(df['dir_D_eff']))
print('\nTop 5 rows:')
print(df.head().to_string(index=False))
print('\nBottom 5 rows:')
print(df.tail().to_string(index=False))
