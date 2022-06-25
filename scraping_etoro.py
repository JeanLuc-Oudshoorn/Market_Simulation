# Imports
import scrapy
import numpy as np
import pandas as pd

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

  # Second parsing method
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

# Generate dataframe from dictionary, need to orient on index and use transpose to generate NA values
trader_frame = pd.DataFrame.from_dict(trader_dictionary, orient = 'index').T

# Generate an object to reorder values from the scraped table by
reorder_index = np.concatenate((np.arange(5)[::-1], np.arange(5, 17)[::-1], np.arange(17, 29)[::-1], np.arange(29, 41)[::-1],
                         np.arange(41, 53)[::-1], np.arange(53, 65)[::-1], np.arange(65, 77)[::-1],
                         np.arange(77, 89)[::-1], np.arange(89, 101)[::-1], np.arange(101, 107)[::-1]))

# Reorder dataframe and drop obsolete columns
trader_frame['new_index'] = reorder_index
trader_frame = trader_frame.reset_index()
trader_frame = trader_frame.set_index('new_index')
trader_frame = trader_frame.sort_index(ascending = True)
trader_frame = trader_frame.drop(columns=['index'])

print(trader_frame.head(30))