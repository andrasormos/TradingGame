# from random import randint
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import plot, draw, show
# import matplotlib.animation as animation
# import matplotlib.image as mpimg
# from matplotlib.widgets import Button, TextBox
# import os.path
# import sys
# from skimage import draw


from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

ts = TimeSeries(key='S45HPVH3ZGG9U18Z', output_format='pandas', indexing_type='date')
df_AAPL, meta_data = ts.get_intraday(symbol='AAPL',interval='15min', outputsize='full')

df_AAPL = df_AAPL.reset_index()
print(df_AAPL)



df_AAPL.drop("1. open", axis=1, inplace=True)
df_AAPL.drop("2. high", axis=1, inplace=True)
df_AAPL.drop("3. low", axis=1, inplace=True)
df_AAPL.drop("5. volume", axis=1, inplace=True)

df_AAPL["Date"] = df_AAPL["date"]
df_AAPL["Close"] = df_AAPL["4. close"]

df_AAPL.drop("date", axis=1, inplace=True)
df_AAPL.drop("4. close", axis=1, inplace=True)

print(df_AAPL)


df_AAPL.to_csv("./cryptoExtract/AAPL/AAPL_15min.csv", header=True, index=True)

# df_4hourly = pd.DataFrame(columns=["Unix", "Date", "Close", "Volume"])
# #len(df_BTC)
# for i in range(len(df_BTC)):
# 	unix_time = df_BTC["Unix"][i]
#
#
# 	if unix_time % 14400 == 0:
# 		next_state_row = df_BTC.loc[[i]]
# 		df_4hourly = pd.concat([df_4hourly, next_state_row])
#
# df_4hourly = df_4hourly.reset_index(drop=True)
# df_4hourly.to_csv("./cryptoExtract/treatment/raw_BTC_GBP_4_hourly.csv", header=True, index=True)