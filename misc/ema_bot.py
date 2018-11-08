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

# now = 1540795500
# end = 1483228800
# difference = now - end
# iterate = difference / 180000
# print(iterate)

df_BTC = pd.read_csv("./cryptoExtract/raw_BTC_GBP.csv", index_col=0)

# SMA
sma_days_A = 30
sma_days_B = 7

# START DATE
begin_date = 20000

# WALLET FINANCES
fiat_cash_balance = 10000
btc_balance = 0
fullBalance = 10000
trade_amount = 1000

# INIT
current_time = len(df_BTC) - 1
days_since_buy = 9600
sma_A = sma_days_A * 96
sma_B = sma_days_B * 96
#sma_list_A = []
#sma_list_B = []
price_list = []
buy_list = []
sell_list = []

# CURRENT CHART STATE
df_segment = df_BTC.loc[:begin_date]
df_segment = df_segment.iloc[::-1].reset_index(drop=True)

# CURRENT STATE PLUS PAST FOR MA
df_segment_plus_past = df_BTC.loc[:begin_date + sma_A]
df_segment_plus_past = df_segment_plus_past.iloc[::-1].reset_index(drop=True)

sma = df_segment_plus_past["Close"].rolling(window=sma_A, min_periods=0).mean()
sma = sma.shift(periods=((sma_A*-1)), freq=None, axis=0)
ema = df_segment_plus_past["Close"].ewm(span=sma_A, min_periods=0, adjust=False, ignore_na=False).mean()
ema = ema.shift(periods=((sma_A*-1)), freq=None, axis=0)




fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(111)
ax1.plot(df_segment["Close"], "-", color='b', linewidth=1)
ax1.plot(sma, "-", color='r', linewidth=2, label=("sma" + str(sma_days_A)))
ax1.plot(ema, "-", color='purple', linewidth=2, label=("ema" + str(sma_days_A)))


#ax1.plot(sma_list_B, "-", color='purple', linewidth=1, label=("sma" + str(sma_days_B)))
ax1.legend()
plt.show()

'''
for i in range(begin_date, 0, -1):
	current_btc_price = df_BTC["Close"][i]

	sma_A_prices = df_BTC["Close"][i:i+sma_A]
	current_sma_A_price = sma_A_prices.sum() / sma_A

	sma_B_prices = df_BTC["Close"][i:i+sma_B]
	current_sma_B_price = sma_B_prices.sum() / sma_B

	buy = 0
	sell = 0

	#BUY
	if current_btc_price > current_sma_A_price and days_since_buy > 9600:
		print(df_BTC["Date"][i], " Price:", current_btc_price, " sma_A:", current_sma_A_price)
		print("		BUY BTC NOW")
		days_since_buy = 0
		btc_balance = trade_amount / current_btc_price
		fiat_cash_balance -= trade_amount
		buy = current_btc_price
		print("\n")

	# SELL
	if current_btc_price < current_sma_B_price and btc_balance != 0:
		print(df_BTC["Date"][i], " Price:", current_btc_price, " sma_A:", current_sma_B_price)
		print("		SELL BTC NOW")
		fiat_cash_balance += btc_balance * current_btc_price
		btc_balance = 0
		print("Fiat Balance:", fiat_cash_balance)
		print("\n")
		sell = current_btc_price

	days_since_buy += 1

	price_list.append(current_btc_price)
	sma_list_A.append(current_sma_A_price)
	sma_list_B.append(current_sma_B_price)
	buy_list.append(buy)
	sell_list.append(sell)






fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(111)
ax1.plot(price_list, "-", color='b', linewidth=1)
ax1.plot(sma_list_A, "-", color='y', linewidth=1, label=("sma" + str(sma_days_A)))
ax1.plot(sma_list_B, "-", color='purple', linewidth=1, label=("sma" + str(sma_days_B)))
ax1.plot(buy_list, "o", color='g', markersize=5)
ax1.plot(sell_list, "o", color='r', markersize=5)
ax1.legend()


plt.show()



'''




