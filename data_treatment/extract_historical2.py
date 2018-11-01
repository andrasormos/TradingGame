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
    # df_BTC = pd.DataFrame(columns=["Date", "Close", "Volume"])
    # df_BTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", mode='a', header=True)
    output = client.get_historic_rates(gdax.BTC_GBP, startDate, endDate, granularity=3600)
    output = np.asarray(output)
    date = output[:, [0]]
    close = output[:, [4]]
    volume = output[:, [5]]

    for i in range(len(close)):

        df_date_column = dt.datetime.fromtimestamp(date[i]).strftime('%Y-%m-%d %I-%p')
        df_close_column = int(close[i])
        df_volume_column = int(volume[i])

        df_BTC.loc[BTCcnt] = df_date_column, df_close_column, df_volume_column
        df_BTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", header=True, index=True)
        BTCcnt += 1

    return df_BTC

def extractETH(startDate, endDate):
    global ETHcnt
    # df_ETH = pd.DataFrame(columns=["Date", "Close", "Volume"])
    # df_ETH.to_csv("./cryptoExtract/treatment/raw_ETH_USD.csv", mode='a', header=True)
    output = client.get_historic_rates(gdax.ETH_USD, startDate, endDate, granularity=3600)
    output = np.asarray(output)

    date = output[:, [0]]
    close = output[:, [4]]
    volume = output[:, [5]]

    for i in range(len(close)):
        df_date_column = dt.datetime.fromtimestamp(date[i]).strftime('%Y-%m-%d %I-%p')
        df_close_column = int(close[i])
        df_volume_column = int(volume[i])

        df_ETH.loc[ETHcnt] = df_date_column, df_close_column, df_volume_column
        df_ETH.to_csv("./cryptoExtract/treatment/raw_ETH_USD.csv", header=True, index=True)
        ETHcnt += 1
    return df_ETH

# give human iso timeframe
end_date = "2018-10-29T22:00:00.000Z"

# convert current to unix time
end_epoch = convert_iso_to_unix(end_date)

# how many weeks would you like it to go back?
iterations = 140

epoch_week_list = []
iso_week_list = []
epoch_week_holder = end_epoch
iso_week_holder = convert_unix_to_iso(end_epoch)

for i in range(iterations):
    epoch_week_list.append(epoch_week_holder)
    iso_week_holder = convert_unix_to_iso(epoch_week_holder)
    iso_week_list.append(iso_week_holder)
    epoch_week_holder -= 345600 # 4 days

for i in range(len(iso_week_list)-1):
    endDate = iso_week_list[i]
    startDate = iso_week_list[i+1]
    #print(endDate,startDate)

    df_BTC = extractBTC(startDate, endDate)
    df_ETH = extractETH(startDate, endDate)
    print(df_BTC)
    print(df_ETH)

# ---------------------------------------------------------------------------------------------------------------------
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
