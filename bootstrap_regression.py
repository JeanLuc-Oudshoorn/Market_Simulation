# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read Dataframe with log returns back to perform analysis
trader_frame = pd.read_csv('trader_frame.csv')
trader_frame = trader_frame.set_index('Date')

# Filter to remove NA
filtered_frame = trader_frame[['SP500', 'Reinhardt Gert Coetzee']].loc[trader_frame['Reinhardt Gert Coetzee'].notnull()]

# Calculate the real best fit regression line
real_fit = np.polyfit(filtered_frame['SP500'], filtered_frame['Reinhardt Gert Coetzee'], 2)


# Example of scatterplot of monthly performance combined with polynomial regression line
fig, ax = plt.subplots()

x = np.linspace(start=-0.2, stop=0.2, num=100)
real_y = np.array(real_fit[0]*x**2 + real_fit[1]*x + real_fit[2])

plt.plot(filtered_frame['SP500'], filtered_frame['Reinhardt Gert Coetzee'], marker='.', linestyle='none', alpha=0.4)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.set_xlim((-0.2, 0.2))
ax.set_ylim((-0.2, 0.2))
plt.plot(x, real_y)
plt.xlabel("Monthly log-returns of S&P500")
plt.ylabel("Monthly log-returns of Reinhardt Gert Coetzee")

plt.title("Polynomial Regression of establish Relationship")
plt.grid()
plt.show()

# Draw pairs Bootstrap Sample
inds = filtered_frame.index
chosen_inds = np.random.choice(inds, size=len(filtered_frame['SP500']))

bootstrap_frame = filtered_frame.loc[chosen_inds,:]

bs_fit = np.polyfit(bootstrap_frame['SP500'], bootstrap_frame['Reinhardt Gert Coetzee'], 2)

# Plot bootstrap Regression line
_, ax = plt.subplots()

x = np.linspace(start=-0.2, stop=0.2, num=107)
bs_y = np.array(bs_fit[0]*x**2 + bs_fit[1]*x + bs_fit[2])

plt.plot(bootstrap_frame['SP500'], bootstrap_frame['Reinhardt Gert Coetzee'], marker='.', linestyle='none', alpha=0.4)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.set_xlim((-0.2, 0.2))
ax.set_ylim((-0.2, 0.2))
plt.plot(x, bs_y)
plt.xlabel("Monthly log-returns of S&P500")
plt.ylabel("Monthly log-returns of Reinhardt Gert Coetzee")

plt.title("Polynomial Regression of establish Relationship (Bootstrap Sample)")
plt.grid()
plt.show()




