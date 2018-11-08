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
sma_days_A = 2
sma_days_B = 14
sma_days_C = 10
std_days_A = 60

# START DATE
btc_state_size = 1 # 33000, 63000
start_date = 40000
end_date = start_date + btc_state_size
game_end_date = 41344

# WALLET FINANCES
fiat_cash_balance = 10000
btc_balance = 0
fullBalance = 10000
trade_amount = 1000

# INIT
how_many_days_past_can_buy = 1
days_since_buy = how_many_days_past_can_buy * 30 * 96

ma_a = 24
ma_b = 48
ma_c = 48
std_a_period = std_days_A * 96

sma_list_A = []
sma_list_B = []
sma_list_C = []
std_list_A = []

price_list = []
profit_list = []
buy_list = []
sell_list = []

look_for_buy = False

look_for_sell = False
activate_buy = False
activate_sell = False
do_once = True
intersect_b_cnt = 0
intersect_c_cnt = 0

past_intersect_b_cnt = 0
past_intersect_c_cnt = 0

btc_state = df_BTC.loc[start_date:end_date]
btc_state_ma_a = df_BTC.loc[(start_date - 400):end_date]
btc_state_ma_b = df_BTC.loc[(start_date - 400):end_date]
btc_state_ma_c = df_BTC.loc[(start_date - 400):end_date]

for i in range(end_date, (game_end_date)):
	if i % 500 == 0:
		print("STEP: ", i, "DATE: ", btc_state["Date"][i])

	# ADD NEXT BTC ROW TO STATE AND DELETE PREVIOUS
	next_state_row = df_BTC.loc[[i+1]]

	btc_state = pd.concat([btc_state, next_state_row])
	btc_state = btc_state.drop(btc_state.index[0])
	btc_price_current = btc_state["Close"].iloc[-1]

	# ADD NEXT EMA ROW
	btc_state_ma_a = pd.concat([btc_state_ma_a, next_state_row])
	btc_state_ma_a = btc_state_ma_a.drop(btc_state_ma_a.index[0])
	btc_state_ma_a_applied = talib.DEMA(btc_state_ma_a["Close"], timeperiod=ma_a)
	btc_state_ma_a_applied = btc_state_ma_a_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_a_price_current = btc_state_ma_a_applied.iloc[-1]

	# ADD NEXT EMA ROW
	btc_state_ma_b = pd.concat([btc_state_ma_b, next_state_row])
	btc_state_ma_b = btc_state_ma_b.drop(btc_state_ma_b.index[0])
	btc_state_ma_b_applied = talib.SMA(btc_state_ma_b["Close"], timeperiod=ma_b)
	btc_state_ma_b_applied = btc_state_ma_b_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_b_price_current = btc_state_ma_b_applied.iloc[-1]

	# ADD NEXT EMA ROW
	btc_state_ma_c = pd.concat([btc_state_ma_c, next_state_row])
	btc_state_ma_c = btc_state_ma_c.drop(btc_state_ma_c.index[0])
	btc_state_ma_c_applied = talib.DEMA(btc_state_ma_c["Close"], timeperiod=ma_c)
	btc_state_ma_c_applied = btc_state_ma_c_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_c_price_current = btc_state_ma_c_applied.iloc[-1]

	sma_list_A.append(btc_state_ma_a_price_current)
	sma_list_B.append(btc_state_ma_b_price_current)
	sma_list_C.append(btc_state_ma_c_price_current)

	if 1==2:
		fig = plt.figure(figsize=(12, 10))
		ax1 = fig.add_subplot(111)
		ax1.plot(btc_state["Close"], "-", color='black', linewidth=2)
		ax1.plot(btc_state_ma_a_applied, "-", color='blue', linewidth=2, label=("sma" + str(sma_days_A)))
		ax1.plot(btc_state_ma_b_applied, "-", color='darkgreen', linewidth=2, label=("ema" + str(sma_days_B)))
		ax1.plot(btc_state_ma_c_applied, "-", color='darkred', linewidth=2, label=("sma" + str(sma_days_C)))
		ax1.legend()
		plt.show()

	buy = 0
	sell = 0
	if do_once == True:
		do_once = False
		#past_intersect_b_cnt = len(np.argwhere(np.diff(np.sign(sma_list_B - sma_list_A))).flatten())
		#past_intersect_c_cnt = len(np.argwhere(np.diff(np.sign(sma_list_C - sma_list_A))).flatten())

	if len(sma_list_A) > 2:
		interA = np.asarray(sma_list_A)
		interB = np.asarray(sma_list_B)
		interC = np.asarray(sma_list_C)
		intersect_b_cnt = len(np.argwhere(np.diff(np.sign(interB - interA))).flatten())
		intersect_c_cnt = len(np.argwhere(np.diff(np.sign(interC - interA))).flatten())


	# PRICE PASSED THROUGH BUY CURVE
	if intersect_b_cnt > past_intersect_b_cnt:
		if btc_state_ma_a_price_current > btc_state_ma_b_price_current:
			activate_buy = True
		if btc_state_ma_a_price_current < btc_state_ma_b_price_current:
			activate_buy = False
		past_intersect_b_cnt = intersect_b_cnt

	# PRICE PASSED THROUGH SELL CURVE
	if intersect_c_cnt > past_intersect_c_cnt:
		if btc_state_ma_a_price_current < btc_state_ma_c_price_current:
			activate_sell = True

		if btc_state_ma_a_price_current > btc_state_ma_c_price_current:
			activate_sell = False


		past_intersect_c_cnt = intersect_c_cnt

	# BUY
	if activate_buy: #and days_since_buy > 96:
		days_since_buy = 0
		btc_balance += trade_amount / btc_price_current
		fiat_cash_balance -= trade_amount
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - 10000), 0))
		print(df_BTC["Date"][i], "Profit: ", profit, "BTC:", btc_balance, "BOUGHT")
		buy = btc_price_current
		activate_buy = False

	# SELL
	if activate_sell and btc_balance != 0:
		fiat_cash_balance += btc_balance * btc_price_current
		btc_balance = 0
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - 10000), 0))
		print(df_BTC["Date"][i], "Profit: ", profit, "BTC:", btc_balance, "SOLD")
		sell = btc_price_current
		activate_sell = False


	days_since_buy += 1

	fullBalance = fiat_cash_balance + btc_balance * btc_price_current
	profit = fullBalance - 10000
	profit = int(np.round(profit, 0))

	price_list.append(btc_price_current)

	#std_list_A.append(btc_state_std_a_price_current)

	buy_list.append(buy)
	sell_list.append(sell)
	profit_list.append(profit)


def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	return [float('nan') if x == 0 else x for x in values]


new_buy_list = zero_to_nan(buy_list)
new_sell_list = zero_to_nan(sell_list)

fig = plt.figure(figsize=(19, 10))
ax1 = fig.add_subplot(111)
fig.tight_layout()

ax1.plot(price_list, "-", color='black', linewidth=1)
ax1.plot(sma_list_A, "-", color='blue', linewidth=2, label=("sma" + str(sma_days_A)))
ax1.plot(sma_list_B, "-", color='darkgreen', linewidth=2, label=("sma" + str(sma_days_B)))
ax1.plot(sma_list_C, "-", color='darkred', linewidth=2, label=("sma" + str(sma_days_C)))
# ax1.plot(std_list_A, "-", color='cyan', linewidth=1.5, label="std")
ax1.plot(new_buy_list, "o", color='darkgreen', markersize=10)
ax1.plot(new_sell_list, "o", color='darkred', markersize=10)

xi = [i for i in range(0, len(new_sell_list))]

for i, txt in enumerate(profit_list):
	if txt >= 0:
		ax1.annotate(txt, (xi[i], new_sell_list[i]), size=20, fontweight='bold', color='green')
	else:
		ax1.annotate(txt, (xi[i], new_sell_list[i]), size=20, fontweight='bold', color='red')

# xp = [p for p in range(0, len(new_buy_list))]
# for p, txt in enumerate(profit_list):
# 	if txt >= 0:
# 		ax1.annotate(txt, (xi[p], new_buy_list[p]), size=10, fontweight='bold', color='green')
# 	else:
# 		ax1.annotate(txt, (xi[p], new_buy_list[p]), size=10, fontweight='bold', color='red')

ax1.legend()

plt.show()