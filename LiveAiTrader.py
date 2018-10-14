import gdax
import numpy as np
import dateutil.parser
import datetime
#from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import schedule
import time
from Trader_AI_v003_decider import Predictor


client = gdax.PublicClient()


# GENERATE NEW CSV
df_BTC = pd.DataFrame(columns=["Date", "Close"])
df_ETH = pd.DataFrame(columns=["Date", "Close"])
BTCcnt = 0
ETHCnt = 0

def extractBTC():
    old_csv = pd.read_csv("/home/andras/PycharmProjects/gdaxMarshall/cryptoExtract/latest_BTC_close.csv")
    output = client.get_product_ticker(gdax.BTC_USD)
    close = np.asarray(output)
    close = output.get('price')
    date = output.get('time')
    df_date = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %I-%p')
    df_close = close

    df_BTC.loc[1] = df_date, df_close
    print(df_BTC)

    new_csv = pd.DataFrame(df_BTC, columns=['Date', 'Close']).append(old_csv, ignore_index=True)
    new_csv.to_csv("/home/andras/PycharmProjects/gdaxMarshall/cryptoExtract/latest_BTC_close.csv", index=False)

def extractETH():
    old_csv = pd.read_csv("/home/andras/PycharmProjects/gdaxMarshall/cryptoExtract/latest_ETH_close.csv")
    output = client.get_product_ticker(gdax.ETH_USD)
    close = np.asarray(output)
    close = output.get('price')
    date = output.get('time')
    df_date = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %I-%p')
    df_close = close

    df_ETH.loc[1] = df_date, df_close
    print(df_ETH)

    new_csv = pd.DataFrame(df_ETH, columns=['Date', 'Close']).append(old_csv, ignore_index=True)
    new_csv.to_csv("/home/andras/PycharmProjects/gdaxMarshall/cryptoExtract/latest_ETH_close.csv", index=False)



serverTimeFetch = client.time()
serverISOTime = serverTimeFetch['iso']
serverHumanTime = datetime.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %H:%M:%S')
serverHour = datetime.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%H')
lastServerHour = serverHour
print("last server hour:", lastServerHour)
print("current server hour:", serverHour)

def job():
    global	lastServerHour

    try:
        serverTimeFetch = client.time()
        serverISOTime = serverTimeFetch['iso']

        serverHumanTime = datetime.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime(
            '%Y-%m-%d %H:%M:%S')
        serverHour = datetime.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%H')
        serverMinute = datetime.datetime.strptime(serverISOTime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%M')
        # print("serverMinute", serverMinute)
        if 1 == 1:

            if serverHour != lastServerHour and int(serverMinute) >= 2:
                print("Time:", serverHumanTime, " - Time to fetch data")
                extractBTC()
                extractETH()


                lastServerHour = serverHour

                print("Passing the last 84 hours to ai")
                print("Receiving action")

                action = ChrisMarshall.predictNextHourNow()

                if action == 1:
                    print("Buying BTC for £10")

                if action == 2:
                    print("Sell all BTC")

                if action == 0 or 3:
                    print("We are skipping")




    # then get btc close data at current server time

    # and add it to csv

    # then get action prediction from ai model (which uses the latest csv)
    # as
    # apply action to API 1

    except:
        print("Couldn't fetch server time")





ChrisMarshall = Predictor()


schedule.every(5).seconds.do(job)
while 1:
   schedule.run_pending()
   time.sleep(1)


