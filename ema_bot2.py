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
df_BTC = df_BTC.iloc[::-1].reset_index(drop=True)
#print(df_BTC)

# SMA
sma_days_A = 30
sma_days_B = 30

# START DATE
btc_state_size = 10000
start_date = 40000
end_date = start_date + btc_state_size
game_end_date = 55000


# WALLET FINANCES
fiat_cash_balance = 10000
btc_balance = 0
fullBalance = 10000
trade_amount = 1000

# INIT
days_since_buy = 3880
ma_a = sma_days_A * 96
ma_b = sma_days_B * 96
sma_list_A = []
sma_list_B = []
price_list = []
buy_list = []
sell_list = []

# CURRENT CHART STATE
df_segment = df_BTC.loc[start_date:end_date]


btc_state = df_BTC.loc[start_date:end_date]
btc_state_ma_a = df_BTC.loc[(start_date - ma_a):end_date]
btc_state_ma_b = df_BTC.loc[(start_date - ma_b):end_date]

#'''
#for i in range(0, (len(df_segment)- (1 + btc_state_size))):
for i in range(end_date, (game_end_date)):


	next_state_row = df_BTC.loc[[i+1]]
	btc_state = pd.concat([btc_state, next_state_row])
	btc_state = btc_state.drop(btc_state.index[0])
	btc_price_current = btc_state["Close"].iloc[-1]


	btc_state_ma_a = pd.concat([btc_state_ma_a, next_state_row])
	btc_state_ma_a = btc_state_ma_a.drop(btc_state_ma_a.index[0])
	btc_state_ma_a["Close"] = btc_state_ma_a["Close"].rolling(window=ma_a, min_periods=0).mean()
	btc_state_ma_a_cropped = btc_state_ma_a.loc[btc_state.index[0]:] # crop the beginning using state's range
	btc_state_ma_a_price_current = btc_state_ma_a_cropped["Close"].iloc[-1]

	btc_state_ma_b = pd.concat([btc_state_ma_b, next_state_row])
	btc_state_ma_b = btc_state_ma_b.drop(btc_state_ma_b.index[0])
	btc_state_ma_b["Close"] = btc_state_ma_b["Close"].ewm(span=ma_b, min_periods=0, adjust=False, ignore_na=False).mean()
	btc_state_ma_b_cropped = btc_state_ma_b.loc[btc_state.index[0]:] # crop the beginning using state's range
	btc_state_ma_b_price_current = btc_state_ma_b_cropped["Close"].iloc[-1]

	print("date:", i)

	fig = plt.figure(figsize=(12, 10))
	ax1 = fig.add_subplot(111)
	ax1.plot(btc_state["Close"], "-", color='b', linewidth=2)
	ax1.plot(btc_state_ma_a_cropped["Close"], "-", color='r', linewidth=2, label=("sma" + str(sma_days_A)))
	ax1.plot(btc_state_ma_b_cropped["Close"], "-", color='purple', linewidth=2, label=("ema" + str(sma_days_B)))
	ax1.legend()
	plt.show()


	# buy = 0
	# sell = 0
	#
	# #BUY
	# if btc_price_current > btc_state_ma_a and days_since_buy > 2880:
	# 	print(df_BTC["Date"][i], " Price:", current_btc_price, " ma_a:", current_ma_a_price)
	# 	print("		BUY BTC NOW")
	# 	days_since_buy = 0
	# 	btc_balance = trade_amount / current_btc_price
	# 	fiat_cash_balance -= trade_amount
	# 	buy = current_btc_price
	# 	print("\n")
	#
	# # SELL
	# if current_btc_price < current_ma_b_price and btc_balance != 0:
	# 	print(df_BTC["Date"][i], " Price:", current_btc_price, " ma_b:", current_ma_b_price)
	# 	print("		SELL BTC NOW")
	# 	fiat_cash_balance += btc_balance * current_btc_price
	# 	btc_balance = 0
	# 	print("Fiat Balance:", fiat_cash_balance)
	# 	print("\n")
	# 	sell = current_btc_price

	days_since_buy += 1
	#
	#
	#
	price_list.append(btc_price_current)
	sma_list_A.append(btc_state_ma_a_price_current)
	sma_list_B.append(btc_state_ma_b_price_current)
	# # buy_list.append(buy)
	# # sell_list.append(sell)






fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(111)
ax1.plot(price_list, "-", color='b', linewidth=2)
ax1.plot(sma_list_A, "-", color='r', linewidth=2, label=("sma" + str(sma_days_A)))
ax1.plot(sma_list_B, "-", color='purple', linewidth=2, label=("sma" + str(sma_days_B)))
#ax1.plot(buy_list, "o", color='g', markersize=5)
#ax1.plot(sell_list, "o", color='r', markersize=5)
ax1.legend()


plt.show()



#'''




