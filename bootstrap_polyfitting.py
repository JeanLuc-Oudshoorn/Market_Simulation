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
def draw_bs_pairs_reg(trader = 'Can Zhao', poly_deg = 2, size=1):
    """
    Performs a pairs bootstrap to do a polynomial regression, adjusts the intercept parameter
    when it implies more than +1.5% outperformance versus the market index and saves the fit parameters
    to a dictionary. The key of the dictionary is the number of the current bootstrap & regression iteration.

    :param trader: name of trader's monthly performance to sample from performance dataframe
    :param poly_deg: polynomial degree, 1 for linear, 2 for exponential
    :param size: number of bootstraps and subsequent regressions to perform
    :return: a dictionary with fit parameters for each boostrap regression
    """
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
    """
    Randomly draws periods of consecutive months from real historical data of the market index. Then uses this
    data as input to predict monthly performance of the given trader based on a randomly chosen set of regression
    fit parameters from a fit_parameters dictionary. Performs this whole process for a specified time period and
    calculates the mean monthly return over this entire period. Repeats this simulation as many times as specified
    and generates an output array of mean returns for each simulation.

    :param fit_dict: dictionary of regression fit parameters
    :param market: list of historical market returns
    :param sim_months: period of time to simulate
    :param months_per_draw: amount of consecutive months to draw from the market index in one go
    :param simulations: how many times the simulation over the full time period is repeated
    :param infl_rate: yearly inlfation rate, expressed as one plus a fraction (eg.: 1.03)
    :param trader: name of the trader to perform the simulation for
    :return: - array of mean monthly performance for the entire time period for each simulation,
             - dictionary of full monthly performances of every simulation,
             - number of simulated months,
             - median case scenario simulation mean annual return (expressed as a percentage)
             - best case scenario simulation mean annual return (expressed as a percentage)
             - worst case scenario simulation mean annual return (expressed as a percentage)
    """
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
def calc_metrics(trader, infl_rate = 1.03, start = 50000, bs_reg_fits = 2000, market_sims = 10000,
                 sim_months = 240, months_per_draw = 4):
    """
    Runs the previous two functions in one go and calculates extra performance metrics. Such as,
    the inflation-adjusted portfolio at the end of the time period in the median case scenario,
    best case scenario and worst case scenario. Also calculates the percentage of simulations in
    which the trader would have been a millionaire by the end of the time period, given a certain
    starting capital.

    :param trader: name of trader to predict for
    :param infl_rate: annual inflation rate expressed as a fraction
    :param start: hypothetical starting capital (to calculate chance of becoming a millionaire)
    :param bs_reg_fits: number of regression fits to perform
    :param market_sims: number of full runs to simulate
    :param sim_months: number of months to simulate for each run
    :param months_per_draw: number of consecutive months to draw from market index in one go
    :return: - trader Name
             - median case scenario simulation mean annual return (expressed as a percentage)
             - best case scenario simulation mean annual return (expressed as a percentage)
             - worst case scenario simulation mean annual return (expressed as a percentage)
             - portfolio value at the end of time period in median case scenario
             - portfolio value at the end of time period in best case scenario
             - portfolio value at the end of time period in worst case scenario
             - percent of runs in which trader became a millionaire over time period given start capital
             - dictionary of all full run simulations

    """
    # Perform regression fits on bootstrapped samples
    fits = draw_bs_pairs_reg(trader = trader, size = bs_reg_fits)

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
    """
    Plots portfolio value for a specified number of full runs over a specified time period, adjusted for inflation
    on a monthly basis. Also plots a horizontal starting capital line to easily identify the number of runs where real
    returns dip into negative territory. Provides the option of saving the resulting plot.

    :param full_run_dict: dictionary of full run simulations
    :param trader: trader name
    :param monthly_inf: monthly inflation
    :param sim_months: number of months to simulate
    :param start: starting capital to base calculations on
    :param portfolios: number of runs to plot
    :param save: boolean, provides the choice to save the plot
    """
    # Initialise empty dictionary to hold portfolio values for different runs
    plotting_dict = dict()

    # Calculate portfolio value for each month per run while adjusting for inflation
    for j in range(portfolios):
        portfolio = np.empty(sim_months+1)
        portfolio[0] = start
        for i in np.arange(1, sim_months+1):
            portfolio[i] = portfolio[i-1] * np.exp(full_run_dict[j][i-1])

        for k in np.arange(len(portfolio)):
            portfolio[k] = portfolio[k] / monthly_inf**[k]

        plotting_dict[j] = portfolio

    # Plot a given number of portfolio runs together with a fixed value line for reference
    fig, ax = plt.subplots()
    for i in range(portfolios):
        _ = plt.plot(np.arange(1, sim_months+2), plotting_dict[i], alpha = 0.5, linewidth = 0.5)

    plt.ylim(0, 10000000)
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

# Initialize empty dataframe to hold trader metrics
trader_metrics = pd.DataFrame(data = None)

# Run the main function for each trader in the dataset
for name in trader_frame.columns[:-1]:
    trader, trader_med, trader_max, trader_min, med_portf, max_portf, min_portf, perc_mill, full_run_dict = \
    calc_metrics(trader = name, bs_reg_fits = 1000, market_sims = 1000, sim_months=300)

    output = pd.Series([trader, trader_med, trader_max, trader_min, med_portf, max_portf, min_portf, perc_mill])

    trader_metrics = pd.concat([trader_metrics, output], axis = 1)

    plot_portfolios(full_run_dict, trader, monthly_inf = np.power(1.03, (1/12)),
                    sim_months = 300, save = False, portfolios = 150)

# Clean up the dataframe
trader_metrics = trader_metrics.T
trader_metrics.columns = ['Trader', 'Med Return', 'Max Return', 'Min Return',
                          'Med Portf', 'Max Portf', 'Min Portf', 'Millionaire %']

# Save the resulting dataframe
trader_metrics.to_csv('trader_metrics.csv')