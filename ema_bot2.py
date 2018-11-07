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
import talib

# SIMULATES THE ACTUAL PASSAGE OF TIME IN 15 MINUTES INCREMENTS

plt.style.use('seaborn-darkgrid')

df_BTC = pd.read_csv("./cryptoExtract/raw_BTC_GBP.csv", index_col=0)
df_BTC = df_BTC.iloc[::-1].reset_index(drop=True)

# SMA
sma_days_A = 14
sma_days_B = 2

# START DATE
btc_state_size = 500 # 49000, 62000
start_date = 33000
end_date = start_date + btc_state_size
game_end_date = 40000

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

btc_state = df_BTC.loc[start_date:end_date]
btc_state_ma_a = df_BTC.loc[(start_date - ma_a):end_date]
btc_state_ma_b = df_BTC.loc[(start_date - ma_b):end_date]

print(btc_state)
print(btc_state_ma_b)

for i in range(end_date, (game_end_date)):
	if i % 500 == 0:
		print("STEP: ", i, "DATE: ", btc_state["Date"][i])

	# ADD NEXT BTC ROW TO STATE AND DELETE PREVIOUS
	next_state_row = df_BTC.loc[[i+1]]

	btc_state = pd.concat([btc_state, next_state_row])
	btc_state = btc_state.drop(btc_state.index[0])
	btc_price_current = btc_state["Close"].iloc[-1]

	# ADD NEXT SMA ROW
	btc_state_ma_a = pd.concat([btc_state_ma_a, next_state_row])  # add to last row
	btc_state_ma_a = btc_state_ma_a.drop(btc_state_ma_a.index[0])  # delete first row
	btc_state_ma_a_applied = btc_state_ma_a["Close"].rolling(window=ma_a, min_periods=0).mean()  # apply average
	btc_state_ma_a_applied = btc_state_ma_a_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_a_price_current = btc_state_ma_a_applied.iloc[-1]

	# ADD NEXT EMA ROW
	btc_state_ma_b = pd.concat([btc_state_ma_b, next_state_row])
	btc_state_ma_b = btc_state_ma_b.drop(btc_state_ma_b.index[0])
	btc_state_ma_b_applied = talib.DEMA(btc_state_ma_b["Close"], timeperiod=ma_b)
	btc_state_ma_b_applied = btc_state_ma_b_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_b_price_current = btc_state_ma_b_applied.iloc[-1]


	if 1==1:
		fig = plt.figure(figsize=(12, 10))
		ax1 = fig.add_subplot(111)
		ax1.plot(btc_state["Close"], "-", color='b', linewidth=2)
		ax1.plot(btc_state_ma_a_applied, "-", color='r', linewidth=2, label=("sma" + str(sma_days_A)))
		ax1.plot(btc_state_ma_b_applied, "-", color='purple', linewidth=2, label=("ema" + str(sma_days_B)))
		ax1.legend()
		plt.show()

	buy = 0
	sell = 0
	if days_since_buy > 2880:
		buy_enabled = True


	# BUY
	if btc_state_ma_b_price_current > btc_state_ma_a_price_current and buy_enabled == True:  #and days_since_buy > 2880:
		days_since_buy = 0
		btc_balance += trade_amount / btc_price_current
		fiat_cash_balance -= trade_amount
		buy_enabled = False
		sell_enabled = True
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		print(df_BTC["Date"][i], " Price:", btc_price_current)
		print("						BUY BTC NOW", "Full Balance: ", fullBalance, "BTC:", btc_balance)
		#print("\n")
		buy = btc_price_current

	# SELL
	if btc_state_ma_b_price_current < btc_state_ma_a_price_current and btc_balance != 0:
		fiat_cash_balance += btc_balance * btc_price_current
		btc_balance = 0
		days_since_buy = 0
		buy_enabled = True
		sell_enabled = False
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		print(df_BTC["Date"][i], " Price:", btc_price_current)
		print("						SELL BTC NOW", "Full Balance:", fullBalance)
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

fig = plt.figure(figsize=(19, 10))
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




