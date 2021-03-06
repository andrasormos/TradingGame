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

# IF DEMA 2 CROSSES EMA 14 IT BUYS AND IF IT CROSSES EMA 10 IT SELLS

plt.style.use('seaborn-darkgrid')

df_BTC = pd.read_csv("./cryptoExtract/raw_BTC_GBP.csv", index_col=0)
#df_BTC = pd.read_csv("./cryptoExtract/raw_ETH_USD.csv", index_col=0)
df_BTC = df_BTC.iloc[::-1].reset_index(drop=True)


# SMA
sma_days_A = 2
sma_days_B = 14
sma_days_C = 10
std_days_A = 60

# START DATE
btc_state_size = 1 # 33000, 63000
start_date = 39000
end_date = start_date + btc_state_size
game_end_date = 40344

# WALLET FINANCES
fiat_cash_balance = 10000
btc_balance = 0
fullBalance = 10000
trade_amount = 1000

# INIT
how_many_days_past_can_buy = 1
days_since_buy = how_many_days_past_can_buy * 30 * 96

ma_a = 48
ma_b = 200
ma_c = 200
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



# df_BTC["MA_A"] = df_BTC["Close"].rolling(window=ma_a, min_periods=0).mean()
# df_BTC["MA_B"] = df_BTC["Close"].rolling(window=ma_b, min_periods=0).mean()

# df_BTC["MA_A"] = df_BTC["Close"].ewm(span=ma_a, min_periods=0, adjust=False, ignore_na=False).mean()
# df_BTC["MA_B"] = df_BTC["Close"].ewm(span=ma_b, min_periods=0, adjust=False, ignore_na=False).mean()

df_BTC["MA_A"] = talib.DEMA(df_BTC["Close"], timeperiod=ma_a)
df_BTC["MA_B"] = talib.SMA(df_BTC["Close"], timeperiod=ma_b)
df_BTC["MA_C"] = talib.DEMA(df_BTC["Close"], timeperiod=ma_c)
df_BTC["STD_A"] = df_BTC["MA_B"].rolling(std_a_period).std()


for i in range(start_date, game_end_date):
	if i % 10000 == 0:
		print("STEP: ", i, "DATE: ", df_BTC["Date"][i], "STD:", btc_state_std_a_price_current)
	btc_price_current = df_BTC["Close"][i]
	btc_state_ma_a_price_current = df_BTC["MA_A"][i]
	btc_state_ma_b_price_current = df_BTC["MA_B"][i]
	btc_state_ma_c_price_current = df_BTC["MA_C"][i]
	btc_state_std_a_price_current = df_BTC["STD_A"][i]

	buy = 0
	sell = 0
	if do_once == True:
		if btc_state_ma_a_price_current < btc_state_ma_b_price_current:
			look_for_buy = True

		if btc_state_ma_a_price_current > btc_state_ma_b_price_current:
			look_for_sell = True

		do_once = False


	if btc_state_ma_a_price_current > btc_state_ma_b_price_current and look_for_buy == True: #and days_since_buy > 384:
		activate_buy = True

	if btc_state_ma_a_price_current < btc_state_ma_b_price_current:
		look_for_buy = True
		activate_buy = False

	if btc_state_ma_a_price_current < btc_state_ma_c_price_current and look_for_sell == True: #and days_since_buy > 384:
		activate_sell = True

	if btc_state_ma_a_price_current > btc_state_ma_c_price_current:
		look_for_sell = True
		activate_sell = False

	# BUY
	if activate_buy: # and days_since_buy > 384:
		days_since_buy = 0
		btc_balance += trade_amount / btc_price_current
		fiat_cash_balance -= trade_amount
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - 10000), 0))
		print(df_BTC["Date"][i], "Profit: ", profit, "BTC:", btc_balance, "BOUGHT")
		buy = btc_price_current

		activate_buy = False
		look_for_buy = False

	# SELL
	if activate_sell and btc_balance != 0:
		fiat_cash_balance += btc_balance * btc_price_current
		btc_balance = 0
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - 10000), 0))
		print(df_BTC["Date"][i], "Profit: ", profit, "BTC:", btc_balance, "SOLD")
		sell = btc_price_current

		activate_sell = False
		look_for_sell = False

	days_since_buy += 1

	fullBalance = fiat_cash_balance + btc_balance * btc_price_current
	profit = fullBalance - 10000
	profit = int(np.round(profit, 0))

	price_list.append(btc_price_current)
	sma_list_A.append(btc_state_ma_a_price_current)
	sma_list_B.append(btc_state_ma_b_price_current)
	sma_list_C.append(btc_state_ma_c_price_current)
	std_list_A.append(btc_state_std_a_price_current)

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
fig.tight_layout()

ax1.plot(price_list, "-", color='black', linewidth=1)
ax1.plot(sma_list_A, "-", color='blue', linewidth=2, label=("sma" + str(sma_days_A)))
ax1.plot(sma_list_B, "-", color='darkgreen', linewidth=2, label=("sma" + str(sma_days_B)))
ax1.plot(sma_list_C, "-", color='darkred', linewidth=2, label=("sma" + str(sma_days_C)))
#ax1.plot(std_list_A, "-", color='cyan', linewidth=1.5, label="std")
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


