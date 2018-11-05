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

# 60		1 minute
# 3600		1 hour
# 86400		1 day
# 604800	1 week
# 2629744	1 month (30.4369 days)
# 31556926	1 year (365.2422 days)

client = gdax.PublicClient()

# GENERATE NEW CSV
df_BTC = pd.DataFrame(columns=["Unix", "Date", "Close", "Volume"])
df_ETH = pd.DataFrame(columns=["Unix", "Date", "Close", "Volume"])
df_compare = pd.DataFrame(columns=["Unix", "Date"])

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
    output = np.asarray(output)

    date = output[:, [0]]
    close = output[:, [4]]
    volume = output[:, [5]]

    for o in range(len(close)):

        df_date_column = dt.datetime.fromtimestamp(date[o]).strftime('%Y-%m-%d %I-%M-%p')
        df_date_inUnix_column = str(int(date[o]))
        df_close_column = int(close[o])
        df_volume_column = int(volume[o])

        df_BTC.loc[BTCcnt] = df_date_inUnix_column, df_date_column, df_close_column, df_volume_column

        BTCcnt += 1

    return df_BTC

def dateFiller(df, compare_list):
    keep_filling = True
    compare_date = int(df["Unix"][0])
    date_index = 0

    while keep_filling == True:
        compare_date -= 900

        if compare_date != int(df["Unix"][date_index])
            print(compare_date, "!=", int(df["Unix"][date_index])


        date_index += 1



    for u in range(len(compare_list)-1):
        extracted = int(df["Unix"][u])
        compare = compare_list[u]

        if extracted != compare:
            print(extracted, "!=", compare)

            A = df[:u].reset_index(drop=True)
            #A.loc[-1] = self.rewardSum, self.profit,

            B = df[u:].reset_index(drop=True)

            print("printing A")
            print(A)
            print("\n")
            print("printing B")
            print(B)
            filled_df = A.append(B).reset_index(drop=True)
            break
    return filled_df


# give human iso timeframe
end_date = "2018-10-29T07:00:00.000Z"
number_of_batches = 4


end_date_unix = convert_iso_to_unix(end_date)
batch_unix_list = []
batch_ISO_list = []
batch_unix = end_date_unix
batch_ISO = convert_unix_to_iso(end_date_unix)

# CREATE DATE COMPARE LIST
date_increment = end_date_unix
for i in range((number_of_batches * 300) - 1):
    if i != 0:
        #compare_unix_list.append(date_increment)
        date_increment_inTimeStamp = dt.datetime.fromtimestamp(date_increment).strftime('%Y-%m-%d %I-%M-%p')
        df_compare.loc[len(df_compare)] = date_increment, date_increment_inTimeStamp

    date_increment -= 900

# CREATE DATE QUERY LIST (ISO DATES 3 DAYS APART (270000 SECONDS) )
for i in range(number_of_batches):
    batch_unix_list.append(batch_unix)
    batch_ISO = convert_unix_to_iso(batch_unix)
    batch_ISO_list.append(batch_ISO)
    batch_unix -= 270000  # 3 days # 270000 = 900×300

# EXTRACT ACTUAL DATA FROM GDAX
for i in range(len(batch_ISO_list)-1):
    endDate = batch_ISO_list[i]
    startDate = batch_ISO_list[i+1]

    df_Extracted_BTC = extractBTC(startDate, endDate)
    sleep(0.4)

# FILL UP MISSING DATES
#filledBTC = dateFiller(df_BTC, compare_unix_list)


print("Compare DF")
print(df_compare)
print("\n")


print("Extracted DF")
print(df_Extracted_BTC)
print("\n")

# print("filledBTC")
# print(filledBTC)




# SAVE THE DF TO A CSV, THIS IS A VERY EFFICIENT METHOD FOR SAVING LOGS AS WELL
filledBTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", header=True, index=True)

