import gdax
import numpy as np
import dateutil.parser
import datetime
#from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

client = gdax.PublicClient()

# GENERATE NEW CSV
df_BTC = pd.DataFrame(columns=["Date", "Close"])
df_ETH = pd.DataFrame(columns=["Date", "Close"])
BTCcnt = 0
ETHCnt = 0

def extractBTC(startDate, endDate):
    global BTCcnt
    output = client.get_historic_rates(gdax.BTC_USD, startDate, endDate, granularity=3600)
    close = np.asarray(output)
    close = close[:, [0, 4]]


    for i in range(len(close)):
        BTCcnt += 1
        unixTime = close[i][0]
        df_humanTime = datetime.datetime.fromtimestamp(unixTime).strftime('%Y-%m-%d %I-%p')
        df_close = close[i][1]

        df_BTC.loc[BTCcnt] = df_humanTime, df_close
        df_BTC.to_csv("./cryptoExtract/latest_BTC_close.csv", index=False)

def extractETH(startDate, endDate):
    global ETHCnt
    output = client.get_historic_rates(gdax.ETH_USD, startDate, endDate, granularity=3600)
    close = np.asarray(output)
    close = close[:, [0, 4]]


    for i in range(len(close)):
        ETHCnt += 1
        unixTime = close[i][0]
        df_humanTime = datetime.datetime.fromtimestamp(unixTime).strftime('%Y-%m-%d %I-%p')
        df_close = close[i][1]

        df_ETH.loc[ETHCnt] = df_humanTime, df_close
        df_ETH.to_csv("./cryptoExtract/latest_ETH_close.csv", index=False)




endDate = "2018-10-14T13:00:00"
startDate = "2018-10-07T20:00:00"
extractBTC(startDate, endDate)
extractETH(startDate, endDate)

endDate = "2018-10-07T20:00:00"
startDate = "2018-10-01T20:00:00"
extractBTC(startDate, endDate)
extractETH(startDate, endDate)

endDate = "2018-10-01T20:00:00"
startDate = "2018-09-25T20:00:00"
extractBTC(startDate, endDate)
extractETH(startDate, endDate)

endDate = "2018-09-25T20:00:00"
startDate = "2018-09-17T20:00:00"
extractBTC(startDate, endDate)
extractETH(startDate, endDate)

endDate = "2018-09-17T20:00:00"
startDate = "2018-09-10T20:00:00"
extractBTC(startDate, endDate)
extractETH(startDate, endDate)



# UNIX TO DATE -------------------------------------------------------------------------
# print("UNIX TO DATE")
# print(datetime.datetime.fromtimestamp(int("1538938800")).strftime('%Y-%m-%d %H:%M:%S'))

# DATE TO UNIX -------------------------------------------------------------------------
# date = datetime.strptime('Sat, 06 Oct 2018 15:00:00', '%a, %d %b %Y %H:%M:%S')
# temp = date.isoformat()
# print("my converted",temp, "\n")






# output = client.time()
# print("time()")
# print(output, "\n")

# output = client.get_currencies()
# print("get_currencies()")
# print(output, "\n")



# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(temp, "-", color='g', linewidth=1)
#
# #plt.axhline(50, color='black', linewidth=0.5)
#
# plt.show()