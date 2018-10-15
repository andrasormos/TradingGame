import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#print(type(df))
#dateParse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
#df = pd.read_csv("Gdax_BTCUSD_1h.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)
df = pd.read_csv("/home/andras/PycharmProjects/TradingGame/crypto/Gdax_BTCUSD_1h_close_latest.csv", index_col=0)

#print(len(df))
#print(df.head())
#print(df.tail())


# trainStartDate = "2017-07-01 11-AM"
# trainEndDate = "2018-08-10 07-AM"
# df_trainData = df.loc[trainEndDate: trainStartDate]
# print("train length",len(df_trainData))
# df_trainData.to_csv("/home/andras/PycharmProjects/TradingGame/crypto/Gdax_ETHUSD_1h_close_train.csv", index=True)




evalStartDate = "2018-08-27 11-AM"
evalEndDate = "2018-10-03 11-AM"



df_evalData = df.loc[evalEndDate: evalStartDate]

df_evalData.to_csv("/home/andras/PycharmProjects/TradingGame/crypto/Gdax_BTCUSD_1h_close_latest_sliced.csv", index=True)
print("eval length",len(df_evalData))





df2 = pd.read_csv("/home/andras/PycharmProjects/TradingGame/crypto/Gdax_ETHUSD_1h_close_latest.csv", index_col=0)


df2_evalData = df2.loc[evalEndDate: evalStartDate]

df2_evalData.to_csv("/home/andras/PycharmProjects/TradingGame/crypto/Gdax_ETHUSD_1h_close_latest_sliced.csv", index=True)
print("eval length",len(df2_evalData))