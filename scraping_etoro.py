# Import packages
import scrapy
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# Import the CrawlerProcess: for running the spider
from scrapy.crawler import CrawlerProcess

# Create the Spider class
class Etoro_Spider(scrapy.Spider):
  name = "etoro_spider"

  # start_requests method
  def start_requests(self):

    start_urls = ["https://factsheets.fundpeak.com/Report/473D3034AE5913E912265730BE689D6D707FA111F2B061DB4F473B892D69F1EAC3B08304018D5E90",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9896E0F6384B3ED10F5C2E5DBF7C6F06EF188F31677E4633FE79D364DFB1B1FE4",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9109B162D8ECFD3118D9D19EEB8B95D54F5B6B838F06055C79FC4C2F0D769E225",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9B18346822950C65D278266233E2B7BEF273197E704D25B2BA3B64565A31953D3",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9B18346822950C65D2225A49B71B9714BB23630CEAE4006E9FC757166B159D96E",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9109B162D8ECFD31182AB71E291E4B04B34CF0742481E80CD4E4A1A2DDDFE8142",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9896E0F6384B3ED10FB41C9C07425485BCAD0B69DECB5826138B63B73C77C9847",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9CB05597DFCB15842C2D8B04E7E7C653943047E140ED621CBB26513E5FA458973",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9B691508D769B17F4FA65D9DFE325EF95D2F6EBF4FE3CE2324322963A738273F9",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E93E82DE55FA2A5F76538DF3A83BED77CBEF2FA39C3597C366AD0605D56D07DD25",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9109B162D8ECFD3118765618479284A81620172FBA30804B2B5BF7A770A15A10E",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9B18346822950C65DB5D196E9CBEA145B4D7248CE5F16967163EA226B72DFB592",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9CB05597DFCB158425A194070CCEFAB3A5BAAAC04897205F77CCD820C8162F0A4",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9896E0F6384B3ED10A5D1C8B25061C87E70CC0E319A51387C05C37EDE5020FCDC",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9896E0F6384B3ED106AC444D5B6E675BDA300E7B2858FA2233356598E13CD0646",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9AE9176B8F5330F049D564864669CA046BB71729B8EB3FA9765B174D7944AC225",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9C30047171818581C070F5FCC81CC1E9AC850B0D2D980B3B140C2B556846A56DF",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9B18346822950C65DB378C89AF3391DB6358F464D5D07E9B627BAF97862968BA8",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E98EB115FA8224524D5DA9755AD76C3D3ADE8BC7078DBADE2B2036981639F6A6FC",
             "https://factsheets.fundpeak.com/Report/473D3034AE5913E9B719965CD9DF91651B126325AF7A0F77FB9620B1BB6096B06CA2246256E499B0"
             ]
    for url in start_urls:
      yield scrapy.Request(url, callback=self.parse)

  # Parse method to extract trader names and their monthly performance
  def parse(self, response):
    # Create a dictionary that contains the name of the trader key and corresponding return values
    name = response.xpath('/html/body/form/div[2]/div[2]/div/div/div/div[2]/div[1]/div[1]/div[2]/div/div/div/text()')
    clean_name = name.get().strip()

    trader_returns = response.xpath('//*[@class = "monthlyPerf"]').re("-?[0-9]+\\.[0-9]{2}")

    trader_dictionary[clean_name] = trader_returns


trader_dictionary = dict()

# Run the Spider
process = CrawlerProcess()
process.crawl(Etoro_Spider)
process.start()

# Generate an object to reorder monthly performance values from the scraped table (they are not chronological)
reorder_index = np.concatenate((np.arange(6)[::-1], np.arange(6, 18)[::-1], np.arange(18, 30)[::-1],
                                np.arange(30, 42)[::-1], np.arange(42, 54)[::-1], np.arange(54, 66)[::-1],
                                np.arange(66, 78)[::-1], np.arange(78, 90)[::-1], np.arange(90, 102)[::-1],
                                np.arange(102, 114)[::-1], np.arange(114, 126)[::-1], np.arange(126, 138)[::-1],
                                np.arange(138, 150)[::-1], np.arange(150, 162)[::-1]))

# Converting the string values in the dictionary to numbers using a nested list comprehension
trader_dictionary = dict([a, [float(i) for i in x]] for a, x in trader_dictionary.items())

# Creating a new dictionary to host the sorted return values
trader_dictionary_sorted = dict()

# Filtering the reordering list to match the length of each monthly performance series
# Sorting monthly performances based on the reordering string above
trader_names = trader_dictionary.keys()

for name in trader_names:
    length = len(trader_dictionary[name]) -1
    reorder_sub = filter(lambda rank: rank <= length, reorder_index)
    reorder_list = list(reorder_sub)

    returns = trader_dictionary[name]
    sorted_returns = [returns[i] for i in reorder_list]
    trader_dictionary_sorted[name] = sorted_returns

# Turning dictionary in a dataframe
# Using "orient = 'index'" and transposing to deal with unequal lengths of monthly performance series
trader_frame = pd.DataFrame.from_dict(trader_dictionary_sorted, orient = 'index').T

# Updating the index of the dataframe
trader_frame = trader_frame.reindex(index = trader_frame.index[::-1])
trader_frame = trader_frame.reset_index(drop=True)
trader_frame = trader_frame.set_index(np.arange(0, 108))

# Converting percentage gain per month into a scalar that can be used to multiply input values
trader_frame = (trader_frame / 100) +1

# Insert a new top row to add a base portfolio value of 1 for the popular investor that has been active the longest
top_row = pd.DataFrame(np.nan, columns = trader_frame.columns, index = [0])
trader_frame = pd.concat([top_row, trader_frame.loc[:]]).reset_index(drop=True)

# For each popular investor, set portfolio value to 1 the month before the first monthly performance is recorded
# Portfolio value for second month becomes the first monthly performance multiplied by the second monthly performance
# ... And so forth
# Values are changed to the natural logarithm of original portfolio values
# Difference between these values per month is taken to end up with a dataframe of log returns
column_names = trader_frame.columns
for name in column_names:
    first_value_to_multiply = trader_frame[name].first_valid_index() + 1
    last_missing = trader_frame[name].first_valid_index() - 1

    trader_frame.loc[last_missing, name] = 1

    for i in np.arange(first_value_to_multiply, len(trader_frame[name])):
        trader_frame.loc[i, name] = trader_frame.loc[i-1, name] * trader_frame.loc[i, name]

    trader_frame[name] = np.log(trader_frame[name])

trader_frame = trader_frame.diff()

# Download S&P500 data from Yahoo Finance
sp = yf.download("^GSPC", start= datetime(1970,6,1), end = datetime(2022,7,1),interval='1mo')

# Select relevant column and convert to log returns
sp = np.log(sp['Adj Close'].dropna())
sp = sp.diff()

# Select relevant time interval
sp_sub = sp['2013-06-01':'2022-06-01']

# Set relevant time interval as index for the dataframe
trader_frame = trader_frame.set_index(sp_sub.index)

# Add the log returns of the S&P500 to the dataframe
trader_frame['SP500'] = sp_sub

# Write dataframe to CSV for further analysis
trader_frame.to_csv('trader_frame.csv')