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
    global extracted_unix_list

    output = client.get_historic_rates(gdax.BTC_GBP, startDate, endDate, granularity=900)
    #print(output)
    output = np.asarray(output)

    date = output[:, [0]]
    close = output[:, [4]]
    volume = output[:, [5]]

    for i in range(len(close)):

        df_date_column = dt.datetime.fromtimestamp(date[i]).strftime('%Y-%m-%d %I-%M-%p')
        df_close_column = int(close[i])
        df_volume_column = int(volume[i])

        df_BTC.loc[BTCcnt] = df_date_column, df_close_column, df_volume_column
        print(str(int(date[i])))
        date_unix = str(int(date[i]))
        extracted_unix_list.append(date_unix)
        #df_BTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", header=True, index=True)
        BTCcnt += 1

    return df_BTC

# 60		1 minute
# 3600		1 hour
# 86400		1 day
# 604800	1 week
# 2629744	1 month (30.4369 days)
# 31556926	1 year (365.2422 days)


# give human iso timeframe
end_date = "2018-10-29T07:00:00.000Z"
end_date_unix = convert_iso_to_unix(end_date)

# how many 300 batches would you like it to go back?
batches = 10
# 1 batch = 300 elements, so in total we have to end yp with 3000 elements

batch_unix_list = []
batch_ISO_list = []
batch_unix = end_date_unix
batch_ISO = convert_unix_to_iso(end_date_unix)

# CREATE LIST OF ISO DATES 270000 SECONDS APART (3 DAYS APART)
for i in range(batches):
    batch_unix_list.append(batch_unix)
    batch_ISO = convert_unix_to_iso(batch_unix)
    batch_ISO_list.append(batch_ISO)
    batch_unix -= 270000  # 3 days # 270000 = 900×300

# CREATE LIST OF 3000 ELEMENTS THAT WILL BE USED TO FIND MISSING ONES IN THE ACTUAL EXTRACTED DATA
compare_unix_list = []
compare_unix = end_date_unix
for i in range(2999):
    compare_unix_list.append(compare_unix)
    compare_unix -= 900

print("compare_unix_list:", len(compare_unix_list))
print(compare_unix_list)
print("\n")

extracted_unix_list = []

# QUERY ACTUAL DATA FROM GDAX
for i in range(len(batch_ISO_list)-1):
    print("Batch:", i)
    endDate = batch_ISO_list[i]
    startDate = batch_ISO_list[i+1]

    df_BTC = extractBTC(startDate, endDate)
    sleep(0.4)

print("extracted_unix_list:", len(extracted_unix_list))
print(extracted_unix_list)
print("\n")

# NOW ITERATE THROUGH THE COMPARE LIST AND COMPARE IT TO THE EXTRACTED DATA AND FILL IN THE MISSING ONES
# IT LOOKS LIKE YOU HAVE TO FILL IN LIKE 310 VALUES AT THE MOMENT

    #df_ETH = extractETH(startDate, endDate)
    #print(df_BTC)
    #print(df_ETH)


# SAVE THE DF TO A CSV, THIS IS A VERY EFFICIENT METHOD FOR SAVING LOGS AS WELL
df_BTC.to_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", header=True, index=True)


