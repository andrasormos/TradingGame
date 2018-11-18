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
from pylab import plot,show
from scipy import interpolate
from numpy import arange
from scipy import ndimage
from scipy import signal

# SIMULATES THE ACTUAL PASSAGE OF TIME IN 15 MINUTES INCREMENTS

# DAY 3 49317 BUY  index: 49318
# DAY 4 49381 SELL index: 49382 Profit: £ -16 Balance: 9984.0 Gain: -16.0
# DAY 4 49399 BUY  index: 49400
# DAY 5 49470 SELL index: 49471 Profit: £ -4 Balance: 9980.0 Gain: -20.0
# DAY 6 49585 BUY  index: 49586
# DAY 7 49635 SELL index: 49636 Profit: £ 3 Balance: 9983.0 Gain: -17.0
# DAY 7 49718 BUY  index: 49719
# DAY 8 49733 BUY  index: 49734
# DAY 8 49739 BUY  index: 49740
# DAY 8 49765 BUY  index: 49766
# DAY 9 49842 SELL index: 49843 Profit: £ 70 Balance: 10053.0 Gain: 53.0
# making graph

df_BTC = pd.read_csv("./cryptoExtract/raw_BTC_GBP.csv", index_col=0)
df_BTC = df_BTC.iloc[::-1].reset_index(drop=True)

# SMA
sma_days_A = 0
sma_days_B = 0
sma_days_C = 0
sma_days_D = 0
std_days_A = 60

# FLASH CRASH TETHER: 59000 - 61000

# 58000 - 60000 SMALLER BUY TRESHOLD PERIOD MIGHT WORK BETTER

# START DATE
btc_state_size = 1000 # 33000, 63000,      39000 - 41344
start_date = 39000
end_date = start_date + btc_state_size
game_end_date = 41000

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
sma_list_A = pd.DataFrame(columns=["Close"])
sma_list_B = pd.DataFrame(columns=["Close"])
sma_list_C = pd.DataFrame(columns=["Close"])
sma_list_D = pd.DataFrame(columns=["Close"])
smooth_list_B = pd.DataFrame(columns=["Close"])
price_list = pd.DataFrame(columns=["Close"])
profit_list = pd.DataFrame(columns=["Close"])
buy_list = pd.DataFrame(columns=["Close"])
sell_list = pd.DataFrame(columns=["Close"])
tan_list_B = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_angle_list_B = pd.DataFrame(columns=["Angle"])

previous_points_plot = pd.DataFrame(columns=["x", "y"])

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

previous_degree = 0
previous_degree = 0

state = df_BTC.loc[start_date:end_date]
state_ma_a = df_BTC.loc[(start_date - ma_a*2 - 1):end_date]
state_ma_b = df_BTC.loc[(start_date - ma_b*2 - 1):end_date]
state_ma_c = df_BTC.loc[(start_date - ma_c*2 - 1):end_date]
state_ma_d = df_BTC.loc[(start_date - ma_d*2 - 1):end_date]
cnt = 0


def calc_tangent(x, y, point_x):
	spline = interpolate.splrep(x, y)
	line = arange(point_x - 15, point_x + 15)
	point_y = interpolate.splev(point_x, spline, der=0)  # f(point_x)
	fprime = interpolate.splev(point_x, spline, der=1)  # f'(point_x)
	tan = point_y + fprime * (line - point_x)  # tangent
	return point_x, point_y, line, tan, fprime


for i in range(end_date, (game_end_date)):
	cnt += 1

	days = int(np.round((cnt / 96), 0))

	next_state_row = df_BTC.loc[[i+1]]
	# PRICE
	state = pd.concat([state, next_state_row])
	state = state.drop(state.index[0])
	btc_price_current = state["Close"].iloc[-1]
	price_list.loc[state.index[-1]] = state.iloc[-1]

	# MA A
	state_ma_a = pd.concat([state_ma_a, next_state_row])
	state_ma_a = state_ma_a.drop(state_ma_a.index[0])
	state_ma_a_applied = talib.DEMA(state_ma_a["Close"], timeperiod=ma_a)
	#state_ma_a_applied = state_ma_a_applied.loc[state.index[0]:]  # crop the beginning using btc state's range
	ma_a_price_current = state_ma_a_applied.iloc[-1]
	sma_list_A.loc[state_ma_a_applied.index[-1]] = state_ma_a_applied.iloc[-1]

	# MA B
	state_ma_b = pd.concat([state_ma_b, next_state_row])
	state_ma_b = state_ma_b.drop(state_ma_b.index[0])
	state_ma_b_applied = talib.SMA(state_ma_b["Close"], timeperiod=ma_b)
	ma_b_price_current = state_ma_b_applied.iloc[-1]
	sma_list_B.loc[state_ma_b_applied.index[-1]] = state_ma_b_applied.iloc[-1]

	# MA C
	state_ma_c = pd.concat([state_ma_c, next_state_row])
	state_ma_c = state_ma_c.drop(state_ma_c.index[0])
	state_ma_c_applied = talib.DEMA(state_ma_c["Close"], timeperiod=ma_c)
	ma_c_price_current = state_ma_c_applied.iloc[-1]
	sma_list_C.loc[state_ma_c_applied.index[-1]] = state_ma_c_applied.iloc[-1]

	# TEST MA
	state_ma_d = pd.concat([state_ma_d, next_state_row])
	state_ma_d = state_ma_d.drop(state_ma_d.index[0])
	state_ma_d_applied = talib.DEMA(state_ma_d["Close"], timeperiod=ma_d)
	ma_d_price_current = state_ma_d_applied.iloc[-1]
	sma_list_D.loc[state_ma_d_applied.index[-1]] = state_ma_d_applied.iloc[-1]

	# GAUSSIAN MA B
	list_y = np.asarray(state_ma_b_applied[-1004:])
	list_x = np.asarray(state_ma_b_applied.index[-1004:])
	df = pd.DataFrame(list_y)
	df = df.rolling(window=7, min_periods=0).mean()
	list_y_gauss = df.values
	y_gauss_current = list_y_gauss[-1]
	smooth_list_B.loc[list_x[-1]] = list_y_gauss[-1]

	if 1==2:
		fig = plt.figure(figsize=(12, 10))
		ax1 = fig.add_subplot(111)
		ax1.plot(state["Close"], "-", color='black', linewidth=2)
		ax1.plot(state_ma_a_applied, "-", color='blue', linewidth=2, label=("sma" + str(sma_days_A)))
		ax1.plot(state_ma_b_applied, "-", color='darkgreen', linewidth=2, label=("ema" + str(sma_days_B)))
		ax1.plot(state_ma_c_applied, "-", color='darkred', linewidth=2, label=("sma" + str(sma_days_C)))
		ax1.plot(list_x, list_y_gauss, "-", color='orange', linewidth=2, label=("buy smooth"))
		#draw_tangent(list_x_gauss, list_y_gauss, list_x_gauss[-1])
		ax1.legend()
		plt.show()

	buy = 0
	sell = 0
	if len(sma_list_A) > 2:
		interPrice = np.asarray(price_list)
		interA = np.asarray(sma_list_A["Close"].values)
		interB = np.asarray(sma_list_B["Close"].values)
		interC = np.asarray(sma_list_C["Close"].values)
		interD = np.asarray(sma_list_D["Close"].values)
		intersect_ab_cnt = len(np.argwhere(np.diff(np.sign(interB - interA))).flatten())
		intersect_ac_cnt = len(np.argwhere(np.diff(np.sign(interC - interA))).flatten())
		intersect_cd_cnt = len(np.argwhere(np.diff(np.sign(interC - interD))).flatten())
		intersect_pc_cnt = len(np.argwhere(np.diff(np.sign(interPrice - interC))).flatten())


	# --------------------------------- CHECK IF SLOPE IS INCREASING ---------------------------------
	point_x, point_y, line, tan, degree = calc_tangent(list_x, list_y_gauss, list_x[-1])
	tan_angle_list_B.loc[state_ma_c_applied.index[-1]] = float(np.round(degree, 3))

	if len(tan_angle_list_B) > 10:
		tan_angle_list_B = tan_angle_list_B.loc[tan_angle_list_B.index[-10]:] # crop off  above 10

		if tan_angle_list_B["Angle"].iloc[-1] > tan_angle_list_B["Angle"].iloc[-10]: # and tan_angle_list_B["Angle"].iloc[-1] >= 0:
			buy_slope_increasing = True
		else:
			buy_slope_increasing = False
	else:
		buy_slope_increasing = False


	# ------------------------- BUY CURVE IS CROSSED UPWARDS -------------------------
	if intersect_ab_cnt > past_intersect_ab_cnt:
		if ma_a_price_current > ma_b_price_current:
			tan_list_B.loc[state_ma_b_applied.index[-1]] = point_x, point_y, line, tan, degree

		if ma_a_price_current > ma_b_price_current and degree > 0.1:
			activate_buy = True
		else:
			activate_buy = False

		past_intersect_ab_cnt = intersect_ab_cnt
	else:
		activate_buy = False
		past_intersect_ab_cnt = intersect_ab_cnt

	# ------------------------- SELL CURVE IS CROSSED DOWNWARDS -------------------------
	if intersect_cd_cnt > past_intersect_cd_cnt and btc_balance != 0: #and degree < 0  #and not buy_slope_increasing:

		if ma_c_price_current <= ma_d_price_current:
			activate_sell = True

		if ma_c_price_current > ma_c_price_current:
			activate_sell = False

		past_intersect_cd_cnt = intersect_cd_cnt

	else:
		activate_sell = False
		past_intersect_cd_cnt = intersect_cd_cnt

	# ------------------------- NEIGHTER HAPPENED -------------------------
	if not activate_buy and not activate_sell:
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = fullBalance - initBalance
		profit = int(np.round(profit, 0))



	# ------------------------- BUY -------------------------
	if activate_buy: # and degree >= 0.1:
		point_x, point_y, line, tan, degree = calc_tangent(list_x, list_y_gauss, list_x[-1])
		try:
			tan_list_B.loc[state_ma_b_applied.index[-1]] = point_x, point_y, line, tan, degree
		except:
			pass

		days_since_buy = 0
		btc_balance += trade_amount / btc_price_current
		fiat_cash_balance -= trade_amount
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - initBalance), 0))
		print("DAY", days, i, "BUY ", "index:", state_ma_a_applied.index[-1])
		#print(df_BTC["Date"][i], "Profit: £", profit, "Balance:",fullBalance, "|BTC:", np.round(btc_balance, 4), "BOUGHT")
		buy = btc_price_current
		activate_buy = False

	# ------------------------- SELL -------------------------
	if activate_sell and btc_balance != 0:
		point_x, point_y, line, tan, degree = calc_tangent(list_x, list_y_gauss, list_x[-1])

		try:
			tan_list_B.loc[state_ma_b_applied.index[-1]] = point_x, point_y, line, tan, degree
		except:
			pass

		fiat_cash_balance += btc_balance * btc_price_current
		btc_balance = 0
		fullBalance = np.round( (fiat_cash_balance + btc_balance * btc_price_current), 0)
		profit = int(np.round((fullBalance - initBalance), 0)) # df_BTC["Date"][i]
		print("DAY", days, i, "SELL", "index:", state_ma_a_applied.index[-1], "Profit: £", profit, "Balance:", fullBalance, "Gain:", (fullBalance- originalBalance))
		#print(cnt, "Profit: £", profit,"Balance:",fullBalance, "|BTC:", np.round(btc_balance, 4), "SOLD")
		sell = btc_price_current
		activate_sell = False
		initBalance = fullBalance



	days_since_buy += 1
	previous_degree = degree

	if buy != 0:
		buy_list.loc[state.index[-1]] = int(buy)
	if sell != 0:
		sell_list.loc[state.index[-1]] = int(sell)
		profit_list.loc[state.index[-1]] = int(profit)


def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	return [float('nan') if x == 0 else x for x in values]

print("making graph")
plt.style.use('dark_background')
fig = plt.figure(figsize=(19, 10))
ax1 = fig.add_subplot(111)
fig.tight_layout()

ax1.plot(price_list, "-", color='gray', linewidth=1.5)
ax1.plot(sma_list_A, "-", color='#86c16a', linewidth=1.5, label=("buy " + str(ma_a)))
ax1.plot(sma_list_B, "-", color='#183515', linewidth=4, label=("buy tresh " + str(ma_b)))
ax1.plot(sma_list_C, "-", color='#bc6454', linewidth=1.5, label=("sell " + str(ma_c)))
ax1.plot(sma_list_D, "-", color='#511308', linewidth=4, label=("sell tresh " + str(ma_d)))
ax1.plot(buy_list["Close"], "o", color='darkgreen', markersize=10)
ax1.plot(sell_list["Close"], "o", color='darkred', markersize=10)

ax1.plot(smooth_list_B.index, smooth_list_B["Close"], "-", color='orange', linewidth=4, label=("buy smooth"), alpha=0.5)

for i in tan_list_B.index:
	if tan_list_B["degree"][i] >= 0:
		ax1.plot(tan_list_B["line"][i], tan_list_B["tan"][i], "--r", color='green')
	else:
		ax1.plot(tan_list_B["line"][i], tan_list_B["tan"][i], "--r", color='red')

ax1.plot(tan_list_B["x"], tan_list_B["y"], "o", color='white', markersize=5)

for i in profit_list["Close"].index:
	text = profit_list["Close"][i]

	if int(text) >= 0:
		ax1.annotate(text, xy=(i, sell_list["Close"][i]), size=20, fontweight='bold', color='green')
	else:
		ax1.annotate(text, xy=(i, sell_list["Close"][i]), size=20, fontweight='bold', color='red')

ax1.legend()
plt.show()
