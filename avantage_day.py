from alphaVantageAPI.alphavantage import AlphaVantage
from random import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox
import os.path
import sys
from skimage import draw



# Initialize the AlphaVantage Class with default values
AV = AlphaVantage(
        api_key="S45HPVH3ZGG9U18Z",
        premium=False,
        output_size='compact',
        datatype='json',
        export=False,
        export_path='~/av_data',
        output='csv',
        clean=False,
        proxy={}
    )

df_AAPL = AV.data(symbol='AAPL', function='D') # Daily

#df_AAPL = AV.intraday(symbol='AAPL', interval='15min') # Intraday as str

df_AAPL.drop("1. open", axis=1, inplace=True)
df_AAPL.drop("2. high", axis=1, inplace=True)
df_AAPL.drop("3. low", axis=1, inplace=True)
df_AAPL.drop("5. volume", axis=1, inplace=True)

df_AAPL["Date"] = df_AAPL["date"]
df_AAPL["Close"] = df_AAPL["4. close"]

df_AAPL.drop("date", axis=1, inplace=True)
df_AAPL.drop("4. close", axis=1, inplace=True)

if 1==1:
    df_filled = pd.DataFrame(columns=["Date", "Close"])
    df_BTC = df_AAPL

    for i in range(len(df_BTC)):
        next_state_row = df_BTC.loc[[i]]
        print(next_state_row)

        for a in range(24):
                df_filled = pd.concat([df_filled, next_state_row])


    #df_filled = df_filled.reset_index(drop=True)
    #df_filled = df_filled.iloc[::-1].reset_index(drop=True)
    df_filled = df_filled.reset_index(drop=True)

    df_filled.to_csv("./cryptoExtract/AAPL/AAPL_15min_filled.csv", header=True, index=True)


print(df_filled)
#df_AAPL.to_csv("./cryptoExtract/AAPL/AAPL_15min.csv", header=True, index=True)
