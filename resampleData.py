import pandas as pd
import numpy as np


dateParse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
df = pd.read_csv("Gdax_BTCUSD_1h.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)
df = df.resample("D").mean()
df_ohlc = df[["Close"]]

#df = df.resample("D", on = "Date").mean()




print(df_ohlc.head())
print("\n")
print(type(df_ohlc))