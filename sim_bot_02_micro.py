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
sma_days_A = 0
sma_days_B = 0
sma_days_C = 0
sma_days_D = 0
std_days_A = 60

# START DATE
btc_state_size = 1 # 33000, 63000,      40000 - 41344
start_date = 40000
end_date = start_date + btc_state_size
game_end_date = 41344

# WALLET FINANCES
fiat_cash_balance = 10000
btc_balance = 0
originalBalance = 10000
initBalance = 10000
fullBalance = 10000
trade_amount = 1000

# INIT
how_many_days_past_can_buy = 1
days_since_buy = how_many_days_past_can_buy * 30 * 96

ma_a = 2 # buy
ma_b = 200 # buy tresh
ma_c = 24 # sell
ma_d = 100 # selltresh

std_a_period = std_days_A * 96
sma_list_A = []
sma_list_B = []
sma_list_C = []
sma_list_D = []
std_list_A = []
price_list = []
profit_list = []
buy_list = []
sell_list = []
look_for_buy = False
look_for_sell = False
activate_buy = False
activate_sell = False
intersect_ab_cnt = 0
intersect_ac_cnt = 0
intersect_cd_cnt = 0
intersect_pc_cnt = 0
past_intersect_ab_cnt = 0
past_intersect_ac_cnt = 0
past_intersect_cd_cnt = 0
past_intersect_pc_cnt = 0

btc_state = df_BTC.loc[start_date:end_date]
btc_state_ma_a = df_BTC.loc[(start_date - ma_a*2 - 1):end_date]
btc_state_ma_b = df_BTC.loc[(start_date - ma_b*2 - 1):end_date]
btc_state_ma_c = df_BTC.loc[(start_date - ma_c*2 - 1):end_date]
btc_state_ma_d = df_BTC.loc[(start_date - ma_d*2 - 1):end_date]
cnt = 0
for i in range(end_date, (game_end_date)):
	cnt+=1
	#if i % 500 == 0:
		#print("STEP: ", i, "DATE: ", btc_state["Date"][i])

	next_state_row = df_BTC.loc[[i+1]]
	# PRICE
	btc_state = pd.concat([btc_state, next_state_row])
	btc_state = btc_state.drop(btc_state.index[0])
	btc_price_current = btc_state["Close"].iloc[-1]

	# MA A
	btc_state_ma_a = pd.concat([btc_state_ma_a, next_state_row])
	btc_state_ma_a = btc_state_ma_a.drop(btc_state_ma_a.index[0])
	btc_state_ma_a_applied = talib.DEMA(btc_state_ma_a["Close"], timeperiod=ma_a)
	#btc_state_ma_a_applied = btc_state_ma_a_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_a_price_current = btc_state_ma_a_applied.iloc[-1]

	# MA B
	btc_state_ma_b = pd.concat([btc_state_ma_b, next_state_row])
	btc_state_ma_b = btc_state_ma_b.drop(btc_state_ma_b.index[0])
	btc_state_ma_b_applied = talib.DEMA(btc_state_ma_b["Close"], timeperiod=ma_b)
	#btc_state_ma_b_applied = btc_state_ma_b_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_b_price_current = btc_state_ma_b_applied.iloc[-1]

	# MA C
	btc_state_ma_c = pd.concat([btc_state_ma_c, next_state_row])
	btc_state_ma_c = btc_state_ma_c.drop(btc_state_ma_c.index[0])
	btc_state_ma_c_applied = talib.DEMA(btc_state_ma_c["Close"], timeperiod=ma_c)
	#btc_state_ma_c_applied = btc_state_ma_c_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_c_price_current = btc_state_ma_c_applied.iloc[-1]

	# TEST MA
	btc_state_ma_d = pd.concat([btc_state_ma_d, next_state_row])
	btc_state_ma_d = btc_state_ma_d.drop(btc_state_ma_d.index[0])
	btc_state_ma_d_applied = talib.DEMA(btc_state_ma_d["Close"], timeperiod=ma_d)
	#btc_state_ma_d_applied = btc_state_ma_d_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_d_price_current = btc_state_ma_d_applied.iloc[-1]

	sma_list_A.append(btc_state_ma_a_price_current)
	sma_list_B.append(btc_state_ma_b_price_current)
	sma_list_C.append(btc_state_ma_c_price_current)
	sma_list_D.append(btc_state_ma_d_price_current)
	price_list.append(btc_price_current)

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
	if len(sma_list_A) > 2:
		interPrice = np.asarray(price_list)
		interA = np.asarray(sma_list_A)
		interB = np.asarray(sma_list_B)
		interC = np.asarray(sma_list_C)
		interD = np.asarray(sma_list_D)
		intersect_ab_cnt = len(np.argwhere(np.diff(np.sign(interB - interA))).flatten())
		intersect_ac_cnt = len(np.argwhere(np.diff(np.sign(interC - interA))).flatten())
		intersect_cd_cnt = len(np.argwhere(np.diff(np.sign(interC - interD))).flatten())
		intersect_pc_cnt = len(np.argwhere(np.diff(np.sign(interPrice - interC))).flatten())


	# CHECK SLOPE
	# print(btc_state_ma_b["Close"])

	ma_b_list = np.asarray(btc_state_ma_b_applied)
	# print(ma_b_list)
	# print(ma_b_list[-1])
	# print(ma_b_list[-5])
	if len(ma_b_list) > 6:
		recentA = ma_b_list[-1]
		pastA = ma_b_list[-2]
		if recentA >= pastA:
			buy_slope_increasing = True
		else:
			buy_slope_increasing = False
	else:
		buy_slope_increasing = False


	# BUY CURVE IS CROSSED UPWARDS
	if intersect_ab_cnt > past_intersect_ab_cnt:
		if btc_state_ma_a_price_current > btc_state_ma_b_price_current and buy_slope_increasing:
			activate_buy = True
		else:
			activate_buy = False
		past_intersect_ab_cnt = intersect_ab_cnt

	# SELL CURVE IS CROSSED DOWNWARDS
	if intersect_cd_cnt > past_intersect_cd_cnt and not buy_slope_increasing:
		if btc_state_ma_c_price_current <= btc_state_ma_d_price_current:
			activate_sell = True

		if btc_state_ma_c_price_current > btc_state_ma_c_price_current:
			activate_sell = False
		past_intersect_cd_cnt = intersect_cd_cnt

	# NEIGHTER HAPPENED
	if not activate_buy and not activate_sell:
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = fullBalance - initBalance
		profit = int(np.round(profit, 0))

	# BUY
	if activate_buy: #and days_since_buy > 96:
		days_since_buy = 0
		btc_balance += trade_amount / btc_price_current
		fiat_cash_balance -= trade_amount
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - initBalance), 0))
		print(cnt, "BUY ")
		#print(df_BTC["Date"][i], "Profit: £", profit, "Balance:",fullBalance, "|BTC:", np.round(btc_balance, 4), "BOUGHT")
		buy = btc_price_current
		activate_buy = False

	# SELL
	if activate_sell and btc_balance != 0:
		fiat_cash_balance += btc_balance * btc_price_current
		btc_balance = 0
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - initBalance), 0)) # df_BTC["Date"][i]
		print(cnt, "Profit: £", profit,"Balance:",fullBalance, "|BTC:", np.round(btc_balance, 4), "SOLD")
		sell = btc_price_current
		activate_sell = False
		initBalance = fullBalance

	days_since_buy += 1

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

ax1.plot(price_list, "-", color='gray', linewidth=1.5)
ax1.plot(sma_list_A, "-", color='lightgreen', linewidth=2, label=("sma" + str(sma_days_A)))
ax1.plot(sma_list_B, "-", color='darkgreen', linewidth=2, label=("sma" + str(sma_days_B)))
ax1.plot(sma_list_C, "-", color='pink', linewidth=2, label=("sma" + str(sma_days_C)))
ax1.plot(sma_list_D, "-", color='darkred', linewidth=2, label=("sma" + str(sma_days_D)))

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