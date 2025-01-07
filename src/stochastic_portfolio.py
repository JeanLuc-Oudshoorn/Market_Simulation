import numpy as np
import pandas as pd
import pymc3 as pm
import cvxpy as cp
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(0)

returns = pd.read_csv('../trader_frame_upd.csv')
returns.set_index('Date', inplace=True)

interest_list = ['^GSPC', 'VGT', 'VTI', 'ASML', 'BRK-B', 'VB']
returns = returns[interest_list].dropna()

n = len(returns.columns)
mean_returns = returns.mean().values
covariance_matrix = returns.cov().values

risk_tolerance = 0.1

# Create PyMC3 model for sampling from the posterior
with pm.Model() as model:
    # Define the variables as probabilistic distributions
    weights = pm.Dirichlet('weights', a=np.ones(mean_returns.shape))
    returns = pm.Normal('returns', mu=mean_returns, sd=1.0, shape=mean_returns.shape)

    # Define the risk as a function of weights and returns
    risk = pm.math.sqrt(pm.math.dot(weights.T, pm.math.dot(covariance_matrix, weights)))

    # Define the objective as the negative of the portfolio return
    objective = -pm.math.dot(weights, returns)

    # Constrain the risk to be less than or equal to the risk tolerance level
    pm.Potential('risk_constraint', risk <= risk_tolerance)

    # Maximize the objective (portfolio return)
    portfolio_return = pm.find_MAP(obj=objective)

# Extract the optimized weights
optimized_weights = portfolio_return['weights']

# Perform quadratic programming using CVXPY
n_assets = len(mean_returns)
w = cp.Variable(n_assets)
objective = cp.Minimize(cp.quad_form(w, covariance_matrix))
constraints = [w >= 0, cp.sum(w) == 1]
risk_constraint = cp.quad_form(w, covariance_matrix) <= risk_tolerance ** 2
problem = cp.Problem(objective, constraints + [risk_constraint])
problem.solve()

# Extract the optimized weights from CVXPY
optimized_weights_cvxpy = w.value

print("Optimized Weights (PyMC3):")
print(optimized_weights)
print("\nOptimized Weights (CVXPY):")
print(optimized_weights_cvxpy)