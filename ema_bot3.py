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
plt.style.use('seaborn-darkgrid')

df_BTC = pd.read_csv("./cryptoExtract/raw_BTC_GBP.csv", index_col=0)
df_BTC = df_BTC.iloc[::-1].reset_index(drop=True)

# SMA
sma_days_A = 20
sma_days_B = 2

# START DATE
btc_state_size = 1 # 49000, 62000
start_date = 33000
end_date = start_date + btc_state_size
game_end_date = 63000

# WALLET FINANCES
fiat_cash_balance = 10000
btc_balance = 0
fullBalance = 10000
trade_amount = 1000

# INIT
how_many_days_past_can_buy = 1
days_since_buy = how_many_days_past_can_buy * 30 * 96

ma_a = sma_days_A * 96
ma_b = sma_days_B * 96
sma_list_A = []
sma_list_B = []
price_list = []
profit_list = []
buy_list = []
sell_list = []
buy_enabled = True
sell_enabled = True


df_BTC["Sma"] = df_BTC["Close"].ewm(span=ma_a, min_periods=0, adjust=False, ignore_na=False).mean()
df_BTC["Ema"] = df_BTC["Close"].ewm(span=ma_b, min_periods=0, adjust=False, ignore_na=False).mean()

if 1 == 2:
	fig = plt.figure(figsize=(12, 10))
	ax1 = fig.add_subplot(111)
	ax1.plot(df_BTC["Close"], "-", color='b', linewidth=1)
	ax1.plot(df_BTC["Sma"], "-", color='r', linewidth=2, label=("sma" + str(sma_days_A)))
	ax1.plot(df_BTC["Ema"], "-", color='purple', linewidth=2, label=("ema" + str(sma_days_B)))
	ax1.legend()
	plt.show()


#'''
for i in range(start_date, game_end_date):
	if i % 10000 == 0:
		print("STEP: ", i, "DATE: ", df_BTC["Date"][i])

	btc_price_current = df_BTC["Close"][i]

	# ADD NEXT SMA ROW
	btc_state_ma_a_price_current = df_BTC["Sma"][i]

	# ADD NEXT SMA ROW
	btc_state_ma_b_price_current = df_BTC["Ema"][i]


	buy = 0
	sell = 0
	# if days_since_buy > 2880:
	# 	buy_enabled = True


	# BUY
	if btc_state_ma_b_price_current > btc_state_ma_a_price_current and buy_enabled == True:  #and days_since_buy > 2880:
		days_since_buy = 0
		btc_balance = trade_amount / btc_price_current
		fiat_cash_balance -= trade_amount
		buy_enabled = False
		sell_enabled = True

		print(df_BTC["Date"][i], " Price:", btc_price_current)
		print("		BUY BTC NOW")
		#print("\n")
		buy = btc_price_current

	# SELL
	if btc_state_ma_b_price_current < btc_state_ma_a_price_current and btc_balance != 0:
		fiat_cash_balance += btc_balance * btc_price_current
		btc_balance = 0
		buy_enabled = True
		sell_enabled = False

		print(df_BTC["Date"][i], " Price:", btc_price_current)
		print("		SELL BTC NOW")
		print("Fiat Balance:", fiat_cash_balance)
		#print("\n")
		sell = btc_price_current

	days_since_buy += 1

	fullBalance = fiat_cash_balance + btc_balance * btc_price_current
	profit = fullBalance - 10000
	profit = np.round(profit, 0)

	price_list.append(btc_price_current)
	sma_list_A.append(btc_state_ma_a_price_current)
	sma_list_B.append(btc_state_ma_b_price_current)
	buy_list.append(buy)
	sell_list.append(sell)
	profit_list.append(profit)


def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	return [float('nan') if x==0 else x for x in values]


new_buy_list = zero_to_nan(buy_list)
new_sell_list = zero_to_nan(sell_list)





fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(111)

ax1.plot(price_list, "-", color='b', linewidth=1)
ax1.plot(sma_list_A, "-", color='orange', linewidth=2, label=("sma" + str(sma_days_A)))
ax1.plot(sma_list_B, "-", color='purple', linewidth=2, label=("sma" + str(sma_days_B)))
ax1.plot(new_buy_list, "*", color='g', markersize=8)
ax1.plot(new_sell_list, "*", color='r', markersize=8)

xi = [i for i in range(0, len(new_sell_list))]
for i, txt in enumerate(profit_list):
	if txt > 0:
		ax1.annotate(txt, (xi[i], new_sell_list[i]), size=10, fontweight='bold', color='green')
	else:
		ax1.annotate(txt, (xi[i], new_sell_list[i]), size=10, fontweight='bold', color='red')

xp = [p for p in range(0, len(new_buy_list))]
for p, txt in enumerate(profit_list):
	if txt > 0:
		ax1.annotate(txt, (xi[p], new_buy_list[p]), size=10, fontweight='bold', color='green')
	else:
		ax1.annotate(txt, (xi[p], new_buy_list[p]), size=10, fontweight='bold', color='red')

ax1.legend()


plt.show()



#'''




