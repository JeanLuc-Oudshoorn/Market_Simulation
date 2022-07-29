import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

popreturn = pd.read_csv('trader_frame.csv')

# 1.) Estimate mean monthly return per investor minus two times standard error
mu = {}

trader_frame = popreturn
trader_frame = trader_frame.set_index('Date')
trader_frame[trader_frame >= 0.14] = 0.14

for name in trader_frame.columns[1:]:
     mu[name] = trader_frame[name].mean() - trader_frame[name].std()/np.sqrt(len(trader_frame[name].dropna()))

mu = pd.DataFrame.from_dict(mu, orient='index', columns=['exp_return'])

# 2.) Estimate optimal allocation with convex optimization
mu = mu.loc[['Jeppe Kirk Bonde', 'Harry Stephan Harrison', 'Libor Vasa',
             'Antonio Emanuele Fasciani', 'Blue Screen Media ApS'], 'exp_return']

popreturn = popreturn[['Jeppe Kirk Bonde', 'Harry Stephan Harrison', 'Libor Vasa',
                       'Antonio Emanuele Fasciani', 'Blue Screen Media ApS']]

mean = mu.values
covs = popreturn.cov().values


def solve_problem(mu = mu, popreturn = popreturn, risk_pref = 0.1):
     mean_stock = mean
     cov_stock = covs

     x = cp.Variable(len(mean_stock))

     stock_return = mean_stock @ x
     stock_risk = cp.quad_form(x, cov_stock)

     objective = cp.Maximize(stock_return - risk_pref * stock_risk)
     constraints = [x >= 0, cp.sum(x) == 1]
     prob = cp.Problem(objective=objective, constraints=constraints)
     return prob.solve(), x.value


# 3.) Plot optimal portfolio allocation for each risk preference
steps = np.linspace(0.01, 2, 100)
x_vals = np.zeros((steps.shape[0], 5))
profit = np.zeros(steps.shape[0])
for i, r in enumerate(steps):
     p, xs = solve_problem(mu, popreturn, risk_pref= r)
     x_vals[i, :] = xs
     profit[i] = p

plt.figure(figsize=(12, 4))
tickers = ['Jeppe Kirk Bonde', 'Harry Stephan Harrison',  'Libor Vasa',
                       'Antonio Emanuele Fasciani', 'Blue Screen Media ApS']
for idx, stock in enumerate(tickers):
    plt.plot(steps, x_vals[:, idx], label=stock)
plt.xlabel("risk avoidance")
plt.ylabel("proportion of investment")
plt.legend()
plt.show()
