

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from zipfile import ZipFile, ZIP_DEFLATED

DATA1 = Path("/mnt/data/dataset1 (1).csv")
DATA2 = Path("/mnt/data/dataset2 (1).csv")
OUTDIR = Path("/mnt/data/hit140_outputs"); OUTDIR.mkdir(exist_ok=True)

d1 = pd.read_csv(DATA1)
d2 = pd.read_csv(DATA2)

for c in ['start_time','rat_period_start','rat_period_end','sunset_time']:
    d1[c] = pd.to_datetime(d1[c], errors='coerce')
d2['time'] = pd.to_datetime(d2['time'], errors='coerce')

for c in ['bat_landing_to_food','seconds_after_rat_arrival','risk','reward','hours_after_sunset']:
    d1[c] = pd.to_numeric(d1[c], errors='coerce')
for c in ['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']:
    d2[c] = pd.to_numeric(d2[c], errors='coerce')

def month_to_season(m):
    if m in [6,7,8]: return 'winter'
    if m in [9,10,11]: return 'spring'
    if m in [12,1,2]: return 'summer'
    if m in [3,4,5]: return 'autumn'
    return np.nan

def coerce_month(x):
    try:
        xi = int(x)
        return xi if 1<=xi<=12 else np.nan
    except:
        for fmt in ("%b","%B"):
            try:
                return pd.to_datetime(str(x), format=fmt).month
            except Exception:
                pass
        return np.nan

d1['rat_present_on_landing'] = (
    (d1['start_time'] >= d1['rat_period_start']) & (d1['start_time'] <= d1['rat_period_end'])
).astype('Int64')
d1['month_int'] = d1['month'].apply(coerce_month); d1['season_name'] = d1['month_int'].apply(month_to_season)
d2['month_int'] = d2['month'].apply(coerce_month); d2['season_name'] = d2['month_int'].apply(month_to_season)
d2['rat_presence_rate'] = d2['rat_minutes']/30.0

d1.to_csv("/mnt/data/cleaned_dataset1.csv", index=False)
d2.to_csv("/mnt/data/cleaned_dataset2.csv", index=False)

plt.figure(); d1['risk'].value_counts().sort_index().plot(kind='bar')
plt.title('Risk-taking vs avoidance (Dataset1)'); plt.xlabel('risk'); plt.ylabel('count'); plt.tight_layout()
plt.savefig(OUTDIR/"fig_risk_counts.png", dpi=180); plt.close()

plt.figure(); d1.groupby('season_name')['risk'].mean().plot(kind='bar')
plt.title('Mean risk by season (Dataset1)'); plt.xlabel('season'); plt.ylabel('mean risk'); plt.tight_layout()
plt.savefig(OUTDIR/"fig_risk_by_season.png", dpi=180); plt.close()

plt.figure(); d1.groupby('season_name')['reward'].mean().plot(kind='bar')
plt.title('Mean reward by season (Dataset1)'); plt.xlabel('season'); plt.ylabel('mean reward'); plt.tight_layout()
plt.savefig(OUTDIR/"fig_reward_by_season.png", dpi=180); plt.close()

plt.figure(); d2.plot(x='hours_after_sunset', y='rat_arrival_number', kind='scatter')
plt.title('Rat arrivals vs hours after sunset (Dataset2)'); plt.xlabel('hours after sunset'); plt.ylabel('rat arrivals'); plt.tight_layout()
plt.savefig(OUTDIR/"fig_rat_arrivals_vs_hours.png", dpi=180); plt.close()

ct_risk = pd.crosstab(d1['rat_present_on_landing'], d1['risk'])
ct_reward = pd.crosstab(d1['rat_present_on_landing'], d1['reward'])
ct_risk.to_csv(OUTDIR/"A_contingency_risk.csv")
ct_reward.to_csv(OUTDIR/"A_contingency_reward.csv")

try:
    from scipy import stats
    chi2_risk = stats.chi2_contingency(ct_risk)
    pd.DataFrame({'chi2':[chi2_risk[0]],'p':[chi2_risk[1]],'dof':[chi2_risk[2]]}).to_csv(OUTDIR/"A_chi2_risk.csv", index=False)
    chi2_reward = stats.chi2_contingency(ct_reward)
    pd.DataFrame({'chi2':[chi2_reward[0]],'p':[chi2_reward[1]],'dof':[chi2_reward[2]]}).to_csv(OUTDIR/"A_chi2_reward.csv", index=False)
except Exception as e:
    pass

mA = d1[['risk','reward','rat_present_on_landing','hours_after_sunset','season_name']].dropna().copy()
mA['rat_present_on_landing'] = mA['rat_present_on_landing'].astype(int)
X = pd.get_dummies(mA[['rat_present_on_landing','hours_after_sunset','season_name']], drop_first=True)
X = sm.add_constant(X)

risk_model = sm.Logit(mA['risk'], X).fit_regularized(alpha=1e-4, L1_wt=0.0, disp=False)
reward_model = sm.Logit(mA['reward'], X).fit_regularized(alpha=1e-4, L1_wt=0.0, disp=False)

def or_table(res, cols):
    params = pd.Series(res.params, index=cols)
    se = getattr(res, "bse", None)
    if se is None: se = pd.Series(np.nan, index=cols)
    z = 1.96
    lower = params - z*se; upper = params + z*se
    return pd.DataFrame({'OR': np.exp(params), '2.5%': np.exp(lower), '97.5%': np.exp(upper)})

or_risk = or_table(risk_model, X.columns); or_risk.to_csv(OUTDIR/"A_logit_risk_ORs.csv")
or_reward = or_table(reward_model, X.columns); or_reward.to_csv(OUTDIR/"A_logit_reward_ORs.csv")

b2 = d2.dropna(subset=['rat_arrival_number','season_name','hours_after_sunset']).copy()
mean_ra, var_ra = b2['rat_arrival_number'].mean(), b2['rat_arrival_number'].var()
use_nb = var_ra > mean_ra * 1.5
if use_nb:
    try:
        arrivals_model = smf.glm('rat_arrival_number ~ C(season_name) + hours_after_sunset', data=b2,
                                 family=sm.families.NegativeBinomial()).fit()
        family = "Negative Binomial"
    except Exception:
        arrivals_model = smf.glm('rat_arrival_number ~ C(season_name) + hours_after_sunset', data=b2,
                                 family=sm.families.Poisson()).fit()
        family = "Poisson (fallback)"
else:
    arrivals_model = smf.glm('rat_arrival_number ~ C(season_name) + hours_after_sunset', data=b2,
                             family=sm.families.Poisson()).fit()
    family = "Poisson"

with open(OUTDIR/"B_arrivals_model_family.txt","w") as f: f.write(family)
with open(OUTDIR/"B_arrivals_model_summary.txt","w") as f: f.write(arrivals_model.summary().as_text())
