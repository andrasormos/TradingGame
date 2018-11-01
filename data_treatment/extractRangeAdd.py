import gdax
import numpy as np
import dateutil.parser
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import dateutil.parser as dp

client = gdax.PublicClient()

# GENERATE NEW CSV
df_BTC = pd.DataFrame(columns=["Date", "Close", "Volume"])
df_ETH = pd.DataFrame(columns=["Date", "Close", "Volume"])
df_BTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", mode='a', header=True)
df_ETH.to_csv("./cryptoExtract/treatment/raw_ETH_USD.csv", mode='a', header=True)

BTCcnt = 0
ETHcnt = 0

def convert_iso_to_unix(t):
    parsed_t = dp.parse(t)
    t_in_seconds = parsed_t.strftime('%s')
    return int(t_in_seconds)

def convert_unix_to_iso(t):
    return datetime.fromtimestamp(t).isoformat()

def extractBTC(startDate, endDate):
    global BTCcnt

    output = client.get_historic_rates(gdax.BTC_GBP, startDate, endDate, granularity=3600)
    print(output)
    output = np.asarray(output)

    date = output[:, [0]]
    close = output[:, [4]]
    volume = output[:, [5]]

    date = date[::-1]
    close = close[::-1]
    volume = volume[::-1]

    for i in range(len(close)):
        df_date_column = dt.datetime.fromtimestamp(date[i]).strftime('%Y-%m-%d %I-%p')
        df_close_column = int(float(close[i]))
        df_volume_column = int(float(volume[i]))

        old_csv = pd.read_csv("./cryptoExtract/live/live_BTC_GBP.csv", index_col=0)
        old_csv.Date = old_csv.Date.shift(1)
        old_csv.Close = old_csv.Close.shift(1)
        old_csv.Volume = old_csv.Volume.shift(1)

        old_csv.loc[0] = df_date_column, df_close_column, df_volume_column
        old_csv.Close = old_csv.Close.astype(int)
        old_csv.Volume = old_csv.Volume.astype(int)
        old_csv.to_csv("./cryptoExtract/live/live_BTC_GBP.csv", index=True)


    return df_BTC



def extractETH(startDate, endDate):
    global ETHcnt

    output = client.get_historic_rates(gdax.ETH_USD, startDate, endDate, granularity=3600)
    output = np.asarray(output)

    date = output[:, [0]]
    close = output[:, [4]]
    volume = output[:, [5]]

    date = date[::-1]
    close = close[::-1]
    volume = volume[::-1]

    for i in range(len(close)):
        df_date_column = dt.datetime.fromtimestamp(date[i]).strftime('%Y-%m-%d %I-%p')
        df_close_column = int(float(close[i]))
        df_volume_column = int(float(volume[i]))

        old_csv = pd.read_csv("./cryptoExtract/live/live_ETH_USD.csv", index_col=0)
        old_csv.Date = old_csv.Date.shift(1)
        old_csv.Close = old_csv.Close.shift(1)
        old_csv.Volume = old_csv.Volume.shift(1)
        old_csv.loc[0] = df_date_column, df_close_column, df_volume_column

        old_csv.Close = old_csv.Close.astype(int)
        old_csv.Volume = old_csv.Volume.astype(int)
        old_csv.to_csv("./cryptoExtract/live/live_ETH_USD.csv", index=True)

    return df_ETH


# give human iso timeframe
end_date = "2018-10-31T08:00:00.000Z"  # ADD ONE MORE TO THIS
start_date = "2018-10-29T22:00:00.000Z"  # THIS ONE IS INCLUDED

extractBTC(start_date, end_date)
extractETH(start_date, end_date)

#
# # ---------------------------------------------------------------------------------------------------------------------
# df = pd.read_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", index_col=0)
# print("total:", len(df))
# eval_df = df[0:3000]
# eval_df.to_csv("./cryptoExtract/trainEval/eval_BTC_GBP.csv", index=True)
# print(eval_df)
#
# df = pd.read_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", index_col=0)
# train_df = df[3000:].reset_index(drop=True)
# train_df.to_csv("./cryptoExtract/trainEval/train_BTC_GBP.csv", index=True)
# print(train_df)
#
#
#
#
# df = pd.read_csv("./cryptoExtract/treatment/raw_ETH_USD.csv", index_col=0)
# print("total:", len(df))
# eval_df = df[0:3000]
# eval_df.to_csv("./cryptoExtract/trainEval/eval_ETH_USD.csv", index=True)
# print(eval_df)
#
# df = pd.read_csv("./cryptoExtract/treatment/raw_ETH_USD.csv", index_col=0)
# train_df = df[3000:].reset_index(drop=True)
# train_df.to_csv("./cryptoExtract/trainEval/train_ETH_USD.csv", index=True)
# print(train_df)
