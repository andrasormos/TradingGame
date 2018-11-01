import gdax
import numpy as np
import datetime as dt
from datetime import datetime
import pandas as pd
import schedule
import time
import dateutil.parser as dp
from TA_078_decider import Predictor

client = gdax.PublicClient()

def convert_iso_to_unix(t):
    parsed_t = dp.parse(t)
    t_in_seconds = parsed_t.strftime('%s')
    return int(t_in_seconds)

def convert_unix_to_iso(t):
    return datetime.fromtimestamp(t).isoformat()

# GENERATE NEW CSV
df_BTC = pd.DataFrame(columns=["Date", "Close", "Volume"])
df_ETH = pd.DataFrame(columns=["Date", "Close", "Volume"])

def extractBTC():
    old_csv = pd.read_csv("./cryptoExtract/live/live_BTC_GBP.csv", index_col=0)
    output = client.get_product_ticker(gdax.BTC_GBP)
    df_date_column = output.get('time')
    inISO = convert_iso_to_unix(df_date_column)
    inISO = inISO - 3600
    df_date_column = convert_unix_to_iso(inISO)
    #df_date_column = dt.datetime.strptime(df_date_column, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %I-%p')
    df_date_column = dt.datetime.strptime(df_date_column, "%Y-%m-%dT%H:%M:%S").strftime('%Y-%m-%d %I-%p')
    df_close_column = int(float(output.get('price')))
    df_volume_column = int(float(output.get('volume')))
    old_csv.Date = old_csv.Date.shift(1)
    old_csv.Close = old_csv.Close.shift(1)
    old_csv.Volume = old_csv.Volume.shift(1)
    old_csv.loc[0] = df_date_column, df_close_column, df_volume_column
    old_csv.Close = old_csv.Close.astype(int)
    old_csv.Volume = old_csv.Volume.astype(int)
    old_csv.to_csv("./cryptoExtract/live/live_BTC_GBP.csv", index=True)

def extractETH():
    old_csv = pd.read_csv("./cryptoExtract/live/live_ETH_USD.csv", index_col=0)
    output = client.get_product_ticker(gdax.ETH_USD)
    df_date_column = output.get('time')
    inISO = convert_iso_to_unix(df_date_column)
    inISO = inISO - 3600
    df_date_column = convert_unix_to_iso(inISO)
    #df_date_column = dt.datetime.strptime(df_date_column, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %I-%p')
    df_date_column = dt.datetime.strptime(df_date_column, "%Y-%m-%dT%H:%M:%S").strftime('%Y-%m-%d %I-%p')
    df_close_column = int(float(output.get('price')))
    df_volume_column = int(float(output.get('volume')))
    old_csv.Date = old_csv.Date.shift(1)
    old_csv.Close = old_csv.Close.shift(1)
    old_csv.Volume = old_csv.Volume.shift(1)
    old_csv.loc[0] = df_date_column, df_close_column, df_volume_column
    old_csv.Close = old_csv.Close.astype(int)
    old_csv.Volume = old_csv.Volume.astype(int)
    old_csv.to_csv("./cryptoExtract/live/live_ETH_USD.csv", index=True)

#'''

serverTimeFetch = client.time()
serverISOTime = serverTimeFetch['iso']
serverHumanTime = dt.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %H:%M:%S')
serverHour = dt.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%H')
lastServerHour = serverHour
print("last server hour:", lastServerHour)
print("current server hour:", serverHour)

def job():
    global lastServerHour

    try:
        serverTimeFetch = client.time()

        serverISOTime = serverTimeFetch['iso']
        serverHumanTime = dt.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %H:%M:%S')
        serverHour = dt.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%H')
        serverMinute = dt.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%M')

        if 1 == 1:
            if serverHour != lastServerHour and int(serverMinute) >= 1:
                print("Time:", serverHumanTime, " - Fetching data!")

                extractBTC()
                extractETH()
                print("Data fetched!")

                lastServerHour = serverHour

                action = ChrisMarshall.predictNextHourNow()
                #action = 1

                if action == 1:
                    print("Buy")

                if action == 2:
                    print("Sell")

                if action == 0 or 3:
                    print("We are skipping")
    except:
        print("Couldn't fetch server time")





ChrisMarshall = Predictor()


schedule.every(30).seconds.do(job)
while 1:
   schedule.run_pending()
   time.sleep(1)

#'''
