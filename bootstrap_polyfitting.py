# Imports
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# Read Dataframe with log returns back to perform analysis
trader_frame = pd.read_csv('trader_frame.csv')
trader_frame = trader_frame.set_index('Date')

# Cap positive outliers at +15% monthly gain
trader_frame[trader_frame >= 0.15] = 0.15


# Download S&P500 data from Yahoo Finance
sp = yf.download("^GSPC", start = datetime(1970, 2, 1), end = datetime(2022, 6, 1),interval='1mo')


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
        # Perform polynomial regression on bootstrap sample
        params = np.array(np.polyfit(bs_x, bs_y, poly_deg))

        # Scale down any excess outperformance if the intercept indicates
        # more than 1.5% monthly outperformance vs. market
        if params[2] >= 0.015:
            params[2] = ((params[2]-0.015)*0.7)+0.015

        fit_parameters[i] = params
    return fit_parameters


# Define function to simulate monthly performances over a given period
def market_simulation(fit_dict, market, sim_months = 240, months_per_draw = 4, simulations = 1000, infl_rate = 1.03,
                      trader = 'Can Zhao'):

    twenty_year_avgs = np.empty(simulations)

    months = np.empty(sim_months)

    keys = list(fit_dict.keys())

    full_run_dict = dict()

    # Run full simulation
    for j in range(simulations):

        # Randomly draw regression fits for the entire simulation period
        drawn_fits = np.random.choice(keys, sim_months)

        drawn_returns = pd.Series(dtype='float')

        # While loop to construct a series of four consecutive months for each randomly chosen index
        # This is to maintain the serial correlation that is present in market index returns
        while len(drawn_returns) <= sim_months:
            ind = np.random.choice(np.arange(len(market)))

            drawn_returns = pd.concat([drawn_returns, market[ind:(ind+months_per_draw)]])

        # For each simulated month make a prediction with random regression fit parameters
        # and a random historical month of the market index as input
        for i in range(sim_months):
            pred = np.poly1d(fit_dict[drawn_fits[i]])
            res = pred(drawn_returns[i])
            months[i] = res

        # Construct a dictionary containing all months of every run
        full_run_dict[j] = list(months)
        mean_monthly = np.mean(months)

        # Save average return for each simulated series of months
        twenty_year_avgs[j] = mean_monthly

    # Convert monthly mean returns to annual percent returns
    full_year_avgs = (np.exp(twenty_year_avgs)-1)*12*100

    # Calculate monthly inflation
    annual_infl = (infl_rate-1)*100
    monthly_infl = np.power(infl_rate, (1/12))

    # Calculate percent of runs with a mean positive monthly return after adjusting for inflation
    perc_positive = np.sum(np.exp(twenty_year_avgs) >= monthly_infl)/len(twenty_year_avgs)*100

    # Calculate median, best and worst average returns over a full run
    med_avg = np.median(full_year_avgs)
    min_avg = np.min(full_year_avgs)
    max_avg = np.max(full_year_avgs)

    print(f"{trader} had an average positive real return over {sim_months} months in {perc_positive:.2f}% "
          f"of simulations, after taking into account average annual inflation of {annual_infl:.1f}%."\
          .format(trader=trader, sim_months = sim_months, perc_positive = perc_positive, annual_infl = annual_infl))

    print(f"Median annual return for all simulations:{med_avg:.1f}%, minimum:{min_avg:.1f}%, maximum:{max_avg:.1f}%. "
          f"(Before inflation)"\
          .format(med_avg=med_avg, min_avg=min_avg, max_avg=max_avg))

    return twenty_year_avgs, full_run_dict, sim_months, med_avg, min_avg, max_avg


# Wrapper function that performs regression fits, market simulation and records the output for a given trader
def calc_metrics(trader, infl_rate = 1.03, start = 50000, bs_linreg_fits = 2000, market_sims = 10000,
                 sim_months = 240, months_per_draw = 4):

    # Perform regression fits on bootstrapped samples
    fits = draw_bs_pairs_linreg(trader = trader, size = bs_linreg_fits)

    # Perform market simulation
    long_avgs, full_run_dict, sim_months, trader_med, trader_min, trader_max =\
    market_simulation(fits, sp, sim_months = sim_months, simulations = market_sims,
                      months_per_draw = months_per_draw, trader = trader, infl_rate=infl_rate)

    # Calculated mean monthly performance necessary to become an inflation-adjusted millionaire over given time horizon
    annual_inf = infl_rate
    monthly_inf = np.power(annual_inf, (1/12))
    start = start
    mill_mean = np.power((1000000/start)*monthly_inf**sim_months, (1/sim_months))

    # Calculate median, maximum and minimum portfolio values for all the simulated runs
    med_portf = int(start*np.median(np.exp(long_avgs))**sim_months / monthly_inf**sim_months)
    max_portf = int(start*np.max(np.exp(long_avgs))**sim_months / monthly_inf**sim_months)
    min_portf = int(start*np.min(np.exp(long_avgs))**sim_months / monthly_inf**sim_months)

    # Calculate chance to become an inflation-adjusted millionaire over given time horizon
    perc_mill = np.sum(np.exp(long_avgs) >= mill_mean)/len(long_avgs)*100

    print(f"{trader} had a 3% annual inflation-adjusted maximum portfolio value of: ${max_portf:d}, "
          f"a median value of ${med_portf:d}, and a minimum value of ${min_portf:d}."\
          .format(trader=trader, max_portf=max_portf, med_portf=med_portf, min_portf=min_portf))

    print(f"{trader} had a 3% annual inflation-adjusted chance of becoming a millionaire of {perc_mill:.1f}%, "
          f"after {sim_months:d} months. (With a starting capital of ${start:d})","\n"\
          .format(trader = trader, perc_mill=perc_mill, sim_months=sim_months, start=start))

    return trader, trader_med, trader_max, trader_min, med_portf, max_portf, min_portf, perc_mill, full_run_dict


# Function to generate plots of a selected number of portfolio runs
def plot_portfolios(full_run_dict, trader, monthly_inf, sim_months=240, start=50000, portfolios=200,
                    save = False):

    # Initialise empty dictionary to hold portfolio values for different runs
    plotting_dict = dict()

    for j in range(portfolios):
        portfolio = np.empty(sim_months+1)
        portfolio[0] = start
        for i in np.arange(1, sim_months+1):
            portfolio[i] = portfolio[i-1] * np.exp(full_run_dict[j][i-1])

        for k in np.arange(len(portfolio)):
            portfolio[k] = portfolio[k] / monthly_inf**[k]

        plotting_dict[j] = portfolio


    fig, ax = plt.subplots()
    for i in range(portfolios):
        _ = plt.plot(np.arange(1, sim_months+2), plotting_dict[i], alpha = 0.5, linewidth = 0.5)

    plt.hlines(start, 1, sim_months + 1, color = 'red', linewidth = 0.85)

    plt.suptitle(f"{trader} Portfolio Simulations over {sim_months} Months"\
                 .format(trader=trader, sim_months=sim_months), fontsize = 16)
    plt.title("Starting Capital is $50,000 | Values are adjusted for annual inflation of 3%")
    plt.xlabel("Months")
    plt.ylabel("Portfolio value in $")
    ax.ticklabel_format(style='plain')

    trader_save_name = str(trader).lower().replace(" ", "_")

    plt.draw()
    if save:
        plt.savefig(f'{trader_save_name}_sims.png'.format(trader_save_name=trader_save_name), dpi = 600)

trader_metrics = pd.DataFrame(data = None)


for name in trader_frame.columns[:-1]:
    trader, trader_med, trader_max, trader_min, med_portf, max_portf, min_portf, perc_mill, full_run_dict = \
    calc_metrics(trader = name, bs_linreg_fits = 1000, market_sims = 10000)

    output = pd.Series([trader, trader_med, trader_max, trader_min, med_portf, max_portf, min_portf, perc_mill])

    trader_metrics = pd.concat([trader_metrics, output], axis = 1)

    plot_portfolios(full_run_dict, trader, monthly_inf = np.power(1.03, (1/12)),
                    sim_months = 240, save = True, portfolios = 150)

trader_metrics = trader_metrics.T
trader_metrics.columns = ['Trader', 'Med Return', 'Max Return', 'Min Return',
                          'Med Portf', 'Max Portf', 'Min Portf', 'Millionaire %']

# Save the resulting Dataframe
trader_metrics.to_csv('trader_metrics.csv')