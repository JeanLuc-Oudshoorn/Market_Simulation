# Imports
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# Read Dataframe with log returns back to perform analysis
trader_frame = pd.read_csv('trader_frame.csv')

# Download S&P500 data from Yahoo Finance
sp = yf.download("^GSPC", start = datetime(1970,6,1), end = datetime(2022,6,1),interval='1mo')


# Select relevant column and convert to log returns
sp = np.log(sp['Adj Close'].dropna())
sp = sp.diff().dropna()


# Define function to draw bootstrap replicates and perform polynomial regression
def draw_bs_pairs_linreg(trader = 'Can Zhao', poly_deg = 2, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Filter dataframe for active period of selected trader
    filtered_frame = trader_frame[['SP500', trader]].loc[trader_frame[trader].notnull()]

    # Set up array of indices to sample from: inds
    inds = filtered_frame.index

    # Initialize dictionary to hold fit parameters
    fit_parameters = dict()

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = filtered_frame['SP500'][bs_inds], filtered_frame[trader][bs_inds]
        fit_parameters[i] = np.polyfit(bs_x, bs_y, poly_deg)

    return fit_parameters

def market_simulation(fit_dict, market, sim_months = 240, months_per_draw = 8, simulations = 1000, infl_rate = 1.03):

    twenty_year_avgs = np.empty(simulations)

    months = np.empty(sim_months)

    keys = list(fit_dict.keys())

    full_run_dict = dict()

    for j in range(simulations):

        drawn_fits = np.random.choice(keys, sim_months)

        drawn_returns = pd.Series(dtype='float')

        while len(drawn_returns) <= sim_months:
            ind = np.random.choice(np.arange(len(market)))

            drawn_returns = pd.concat([drawn_returns, market[ind:(ind+months_per_draw)]])

        for i in range(sim_months):
            pred = np.poly1d(fit_dict[drawn_fits[i]])
            res = pred(drawn_returns[i])
            months[i] = res

        full_run_dict[j] = list(months)
        mean_monthly = np.mean(months)
        twenty_year_avgs[j] = mean_monthly

    full_year_avgs = (np.exp(twenty_year_avgs)-1)*12*100

    annual_infl = (infl_rate-1)*100
    monthly_infl = np.power(infl_rate, (1/12))

    perc_positive = np.sum(np.exp(twenty_year_avgs) >= monthly_infl)/len(twenty_year_avgs)*100
    long_avg = np.mean(full_year_avgs)
    min_avg = np.min(full_year_avgs)
    max_avg = np.max(full_year_avgs)

    print(f"Trader had an average positive real return over {sim_months} months in {perc_positive:.3f}% "
          f"of simulations, after taking into account average annual inflation of {annual_infl:.1f}%."\
          .format(sim_months = sim_months, perc_positive = perc_positive, annual_infl = annual_infl))

    print(f"Mean annual return for all simulations:{long_avg:.1f}%, minimum:{min_avg:.1f}%, maximum:{max_avg:.1f}%."\
          .format(long_avg=long_avg, min_avg=min_avg, max_avg=max_avg))

    return twenty_year_avgs, full_run_dict, sim_months, long_avg, min_avg, max_avg


jeppe_fits = draw_bs_pairs_linreg(trader = 'Jeppe Kirk Bonde', size = 2000)

long_avgs, full_run_dict, sim_months, jeppe_mean, jeppe_min, jeppe_max =\
    market_simulation(jeppe_fits, sp, sim_months= 240, simulations = 1000)

annual_inf = 1.03
monthly_inf = np.power(annual_inf, (1/12))
start = 50000

mean_portf = int(start*np.mean(np.exp(long_avgs))**sim_months / monthly_inf**sim_months)
max_portf = int(start*np.max(np.exp(long_avgs))**sim_months / monthly_inf**sim_months)
min_portf = int(start*np.min(np.exp(long_avgs))**sim_months / monthly_inf**sim_months)


print(f"Trader had a 3% annual inflation-adjusted maximum portfolio value of: ${max_portf:d}, "
      f"a mean value of ${mean_portf:d}, and a minimum value of ${min_portf:d}."\
      .format(max_portf=max_portf, mean_portf=mean_portf, min_portf=min_portf))

plotting_dict = dict()

min_ind = np.argmin(long_avgs)
max_ind = np.argmax(long_avgs)

for j in range(200):
    run = full_run_dict[j]
    portfolio = np.empty(sim_months+1)
    portfolio[0] = start
    for i in np.arange(1, sim_months+1):
        portfolio[i] = portfolio[i-1] * np.exp(full_run_dict[j][i-1])

    for k in np.arange(len(portfolio)):
        portfolio[k] = portfolio[k] / monthly_inf**[k]

    plotting_dict[j] = portfolio


fig, ax = plt.subplots()
for i in range(200):
    _ = plt.plot(np.arange(1, sim_months+2), plotting_dict[i], alpha = 0.5, linewidth = 0.5)

plt.hlines(start, 1, sim_months + 1, color = 'red', linewidth = 0.85)

plt.suptitle("Jeppe Kirk Bonde Portfolio Simulations over 20 Years", fontsize = 16)
plt.title("Starting Capital is $50,000 | Values are adjusted for annual inflation of 3%")
plt.xlabel("Months")
plt.ylabel("Portfolio value in $")
ax.ticklabel_format(style='plain')

plt.draw()
plt.savefig('jeppe_sims.png', dpi = 900)
plt.show()
