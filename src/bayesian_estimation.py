# Imports
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

print(f"Running on PyMC3 v{pm.__version__}")
az.style.use("arviz-darkgrid")

# Load Most recent DataFrame for Analysis
trader_frame = pd.read_csv('../trader_frame.csv')
trader_frame = trader_frame.set_index('Date')

sub_frame = (trader_frame[['Jeppe Kirk Bonde', 'SP500']].dropna())

# Define Two Groups
y1 = np.array(sub_frame['Jeppe Kirk Bonde'])
y2 = np.array(sub_frame['SP500'])

y = pd.DataFrame(
    dict(value=np.r_[y1, y2], group=np.r_[["Jeppe Kirk Bonde"] * len(y1), ["SP500"] * len(y2)])
)

#y.hist("value", by="group", figsize=(12, 4))

# Define Means and standard deviations
mu_m = y.value.mean()
mu_s = y.value.std()*2

sigma_low = mu_s / 20
sigma_high = mu_s * 20


with pm.Model() as model:
    group1_mean = pm.Normal('group1_mean', mu=mu_m, sigma=mu_s)
    group2_mean = pm.Normal('group2_mean', mu=mu_m, sigma=mu_s)
    group1_std = pm.Uniform('group1_std', lower=sigma_low, upper=sigma_high)
    group2_std = pm.Uniform('group2_std', lower=sigma_low, upper=sigma_high)

with model:
    v1 = pm.Exponential('v_minus_one1',1/2.0) + 1
    v2 = pm.Exponential('v_minus_one2',1/4.0) + 1
    l1 = group1_std ** -2
    l2 = group2_std ** -2

with model:
    group1 = pm.StudentT('trader', nu=v1, mu=group1_mean, lam=l1, observed=y1)
    group2 = pm.StudentT('SP500', nu=v2, mu=group2_mean, lam=l2, observed=y2)

with model:
    diff_of_means = pm.Deterministic("difference of means", (group1_mean - group2_mean))
    diff_of_stds = pm.Deterministic("difference of stds", (group1_std - group2_std))
    trace = pm.sample(2000, return_inferencedata=True)

# az.plot_posterior(
#     trace,
#     var_names=["difference of means", "difference of stds", "effect size"],
#     ref_val=0,
#     color="#87ceeb",
# )

az.summary(trace, var_names=["difference of means", "difference of stds", "effect size"])



