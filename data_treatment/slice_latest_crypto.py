import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", index_col=0)
print("total:", len(df))
eval_df = df[0:3000]
eval_df.to_csv("./cryptoExtract/trainEval/eval_BTC_GBP.csv", index=True)
eval_df.to_csv("./cryptoExtract/live/live_BTC_GBP.csv", index=True)
print(eval_df)

df = pd.read_csv("./cryptoExtract/treatment/raw_BTC_GBP.csv", index_col=0)
train_df = df[3000:].reset_index(drop=True)
train_df.to_csv("./cryptoExtract/trainEval/train_BTC_GBP.csv", index=True)
print(train_df)




df = pd.read_csv("./cryptoExtract/treatment/raw_ETH_USD.csv", index_col=0)
print("total:", len(df))
eval_df = df[0:3000]
eval_df.to_csv("./cryptoExtract/trainEval/eval_ETH_USD.csv", index=True)
eval_df.to_csv("./cryptoExtract/live/live_ETH_USD.csv", index=True)
print(eval_df)

df = pd.read_csv("./cryptoExtract/treatment/raw_ETH_USD.csv", index_col=0)
train_df = df[3000:].reset_index(drop=True)
train_df.to_csv("./cryptoExtract/trainEval/train_ETH_USD.csv", index=True)
print(train_df)
