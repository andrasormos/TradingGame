import gdax
import numpy as np
import dateutil.parser
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import dateutil.parser as dp
from time import sleep
import os.path


client = gdax.PublicClient()

# GENERATE NEW CSV
df_BTC = pd.DataFrame(columns=["Date", "Close", "Volume"])
df_ETH = pd.DataFrame(columns=["Date", "Close", "Volume"])

if os.path.exists("./cryptoExtract/treatment/raw_BTC_GBP.csv") == True:
    os.remove("./cryptoExtract/treatment/raw_BTC_GBP.csv")
if os.path.exists("./cryptoExtract/treatment/raw_ETH_USD.csv") == True:
    os.remove("./cryptoExtract/treatment/raw_ETH_USD.csv")


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

    output = client.get_historic_rates(gdax.BTC_GBP, startDate, endDate, granularity=900)
    print(output)
    output = np.asarray(output)

    date = output[:, [0]]
    close = output[:, [4]]
    volume = output[:, [5]]

    for i in range(len(close)):

        df_date_column = dt.datetime.fromtimestamp(date[i]).strftime('%Y-%m-%d %I-%M-%p')
        df_close_column = int(close[i])
        df_volume_column = int(volume[i])

        df_BTC.loc[BTCcnt] = df_date_column, df_close_column, df_volume_column
        #df_BTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", header=True, index=True)
        BTCcnt += 1

    return df_BTC

# 60		1 minute
# 3600		1 hour
# 86400		1 day
# 604800	1 week
# 2629744	1 month (30.4369 days)
# 31556926	1 year (365.2422 days)

# extracts 15 minutes increments (900)
# go through data and check if an increment is missing, if missing, just copy previous price or mean between next

# FOR THIS KEEP THE DATE IN UNIX!!!!!!

# save 15min data into csv
# save 60 min data into csv


# give human iso timeframe
end_date = "2018-11-01T07:00:00.000Z"
end_date_unix = convert_iso_to_unix(end_date)

# how many weeks would you like it to go back?
iterations = 10

last300_list = []
last300_ISO_list = []
last300 = end_date_unix
last300_ISO = convert_unix_to_iso(end_date_unix)

for i in range(iterations):
    last300_list.append(last300)
    last300_ISO = convert_unix_to_iso(last300)
    last300_ISO_list.append(last300_ISO)
    last300 -= 270000

#print(last300_list)

for i in range(len(last300_ISO_list)-1):
    print("Iteration:", i)
    endDate = last300_ISO_list[i]
    startDate = last300_ISO_list[i+1]

    df_BTC = extractBTC(startDate, endDate)
    print("sleep")
    sleep(0.4)

    #df_ETH = extractETH(startDate, endDate)
    #print(df_BTC)
    #print(df_ETH)



df_BTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", header=True, index=True)

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
