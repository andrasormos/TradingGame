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
import timeit

# SIMULATES THE ACTUAL PASSAGE OF TIME IN 15 MINUTES INCREMENTS

df_BTC = pd.read_csv("./cryptoExtract/raw_BTC_GBP.csv", index_col=0)
df_BTC = df_BTC.iloc[::-1].reset_index(drop=True)

df_BTC['Unix'] = df_BTC['Unix'].astype('uint32')
df_BTC['Close'] = df_BTC['Close'].astype('uint16')
df_BTC['Volume'] = df_BTC['Volume'].astype('uint8')

#df_BTC['Close'] = df_BTC['Close'].apply(pd.to_numeric, downcast='unsigned')

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
state_start_date_idx = 39000
game_start_date_idx = state_start_date_idx + btc_state_size
game_end_date_idx = 41000

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

ma_a_size = 2 # buy
ma_b_size = 100 # buy tresh
ma_c_size = 24 # sell
ma_d_size = 100 # selltresh

std_a_period = std_days_A * 96

df_ma_charts = pd.DataFrame(columns=["mA", "mB", "mC", "mD", "smoothB"])
df_ma_charts['mA'] = df_ma_charts['mA'].astype('float32')
df_ma_charts['mB'] = df_ma_charts['mB'].astype('float32')
df_ma_charts['mC'] = df_ma_charts['mC'].astype('float32')
df_ma_charts['mD'] = df_ma_charts['mD'].astype('float32')
df_ma_charts['smoothB'] = df_ma_charts['smoothB'].astype('float32')

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

# state = df_BTC.loc[state_start_date_idx:game_start_date_idx]
# state_ma_a = df_BTC.loc[(state_start_date_idx - ma_a_size * 2 - 1):game_start_date_idx]
# state_ma_b = df_BTC.loc[(state_start_date_idx - ma_b_size * 2 - 1):game_start_date_idx]
# state_ma_c = df_BTC.loc[(state_start_date_idx - ma_c_size * 2 - 1):game_start_date_idx]
# state_ma_d = df_BTC.loc[(state_start_date_idx - ma_d_size * 2 - 1):game_start_date_idx]
cnt = 0

def calc_tangent(x, y, point_x):
	spline = interpolate.splrep(x, y)
	line = arange(point_x - 15, point_x + 15)
	point_y = interpolate.splev(point_x, spline, der=0)  # f(point_x)
	fprime = interpolate.splev(point_x, spline, der=1)  # f'(point_x)
	tan = point_y + fprime * (line - point_x)  # tangent
	return point_x, point_y, line, tan, fprime

df_segment = df_BTC.loc[game_start_date_idx:game_end_date_idx]


for index, row in df_segment.iterrows():
	#print(index, row['Unix'], row['Close'])

	cnt += 1
	days = int(np.round((cnt / 96), 0))
	if cnt % 96 == 0:
		print("DAY", days)

	#start_time = timeit.default_timer()
	#state = df_BTC['Close'].loc[-600:index]
	state_ma_a = df_BTC['Close'].loc[index - 600 - ma_a_size * 2: index]
	state_ma_b = df_BTC['Close'].loc[index - 600 - ma_b_size * 2: index]
	state_ma_c = df_BTC['Close'].loc[index - 600 - ma_c_size * 2: index]
	state_ma_d = df_BTC['Close'].loc[index - 600 - ma_d_size * 2: index]

	state_ma_a = np.round( (talib.DEMA(state_ma_a, timeperiod=ma_a_size)), 6 )#.astype('float32')
	state_ma_b = np.round( (talib.DEMA(state_ma_b, timeperiod=ma_b_size)), 6 ).dropna()#.astype('float32')
	state_ma_c = np.round( (talib.DEMA(state_ma_c, timeperiod=ma_c_size)), 6 )#.astype('float32')
	state_ma_d = np.round( (talib.DEMA(state_ma_d, timeperiod=ma_d_size)), 6 )#.astype('float32')

	# SMOOTH
	state_ma_b_smooth_x = np.asarray(state_ma_b.index)
	state_ma_b_smooth = state_ma_b.rolling(window=7, min_periods=0).mean()
	state_ma_b_smooth = state_ma_b_smooth.astype('float32')

	btc_price_current = df_BTC['Close'].loc[index]
	ma_a_price_current = state_ma_a.iloc[-1]
	ma_b_price_current = state_ma_b.iloc[-1]
	ma_c_price_current = state_ma_c.iloc[-1]
	ma_d_price_current = state_ma_d.iloc[-1]
	ma_b_smooth_current = state_ma_b_smooth.iloc[-1]

	price_list.loc[index] = df_BTC['Close'].loc[index]
	price_list = price_list.astype('uint16')

	df_ma_charts.loc[index] = ma_a_price_current, ma_b_price_current, ma_c_price_current, ma_d_price_current, ma_b_smooth_current

	df_ma_charts['mA'] = df_ma_charts['mA'].astype('float32')
	df_ma_charts['mB'] = df_ma_charts['mB'].astype('float32')
	df_ma_charts['mC'] = df_ma_charts['mC'].astype('float32')
	df_ma_charts['mD'] = df_ma_charts['mD'].astype('float32')
	df_ma_charts['smoothB'] = df_ma_charts['smoothB'].astype('float32')

	buy = 0
	sell = 0
	if len(df_ma_charts["mA"]) > 2:
		interPrice = np.asarray(price_list)
		interA = np.asarray(df_ma_charts["mA"].iloc[-2:])
		interB = np.asarray(df_ma_charts["mB"].iloc[-2:])
		interC = np.asarray(df_ma_charts["mC"].iloc[-2:])
		interD = np.asarray(df_ma_charts["mD"].iloc[-2:])

		intersect_ab_cnt = len(np.argwhere(np.diff(np.sign(interB - interA))).flatten())
		intersect_ac_cnt = len(np.argwhere(np.diff(np.sign(interC - interA))).flatten())
		intersect_cd_cnt = len(np.argwhere(np.diff(np.sign(interC - interD))).flatten())
		intersect_pc_cnt = len(np.argwhere(np.diff(np.sign(interPrice - interC))).flatten())

	# --------------------------------- CHECK IF SLOPE IS INCREASING ---------------------------------
	point_x, point_y, line, tan, degree = calc_tangent(state_ma_b_smooth_x, state_ma_b_smooth, state_ma_b_smooth_x[-1])

	start_time = timeit.default_timer()


	tan_angle_list_B.loc[index] = np.round(degree, 3).astype('float16')

	# just afte 11 it becomes slow, but its fast before that
	# something about the chopping of it
	# do the slicing with the current idex minus 1






	print(timeit.default_timer() - start_time)
	#print(tan_angle_list_B)

	if len(tan_angle_list_B) > 10:
		tan_angle_list_B = tan_angle_list_B.loc[tan_angle_list_B.index[-10]:] # crop off  above 10

		if tan_angle_list_B["Angle"].iloc[-1] > tan_angle_list_B["Angle"].iloc[-10]: # and tan_angle_list_B["Angle"].iloc[-1] >= 0:
			buy_slope_increasing = True
		else:
			buy_slope_increasing = False
	else:
		buy_slope_increasing = False

	# ------------------------- BUY CURVE IS CROSSED UPWARDS -------------------------
	if intersect_ab_cnt > 0:
		intersect_ab_cnt = 0
		if ma_a_price_current > ma_b_price_current:
			tan_list_B.loc[state_ma_b.index[-1]] = point_x, point_y, line, tan, degree

		if ma_a_price_current > ma_b_price_current and degree > 0.1:
			activate_buy = True
		else:
			activate_buy = False
	else:
		activate_buy = False

	# ------------------------- SELL CURVE IS CROSSED DOWNWARDS -------------------------
	if intersect_cd_cnt > 0 and btc_balance != 0: #and degree < 0  #and not buy_slope_increasing:
		intersect_cd_cnt = 0
		if ma_c_price_current <= ma_d_price_current:
			activate_sell = True

		if ma_c_price_current > ma_c_price_current:
			activate_sell = False
	else:
		activate_sell = False

	# ------------------------- NEIGHTER HAPPENED -------------------------
	# if not activate_buy and not activate_sell:
	# 	fullBalance = fiat_cash_balance + btc_balance * btc_price_current
	# 	profit = fullBalance - initBalance
	# 	profit = int(np.round(profit, 0))

	# ------------------------- BUY -------------------------
	if activate_buy: # and degree >= 0.1:
		#point_x, point_y, line, tan, degree = calc_tangent(state_ma_b_smooth_x, state_ma_b_smooth, state_ma_b_smooth_x[-1])
		try:
			tan_list_B.loc[state_ma_b_applied.index[-1]] = point_x, point_y, line, tan, degree
		except:
			pass

		days_since_buy = 0
		btc_balance += trade_amount / btc_price_current
		fiat_cash_balance -= trade_amount
		fullBalance = fiat_cash_balance + btc_balance * btc_price_current
		profit = int(np.round((fullBalance - initBalance), 0))
		print("DAY", days, index, "BUY ", "index:", state_ma_a.index[-1])
		#print(df_BTC["Date"][i], "Profit: £", profit, "Balance:",fullBalance, "|BTC:", np.round(btc_balance, 4), "BOUGHT")
		buy = btc_price_current
		activate_buy = False

	# ------------------------- SELL -------------------------
	if activate_sell and btc_balance != 0:
		#point_x, point_y, line, tan, degree = calc_tangent(state_ma_b_smooth_x, state_ma_b_smooth, state_ma_b_smooth_x[-1])

		try:
			tan_list_B.loc[state_ma_b_applied.index[-1]] = point_x, point_y, line, tan, degree
		except:
			pass

		fiat_cash_balance += btc_balance * btc_price_current
		btc_balance = 0
		fullBalance = np.round( (fiat_cash_balance + btc_balance * btc_price_current), 0)
		profit = int(np.round((fullBalance - initBalance), 0)) # df_BTC["Date"][i]
		print("DAY", days, index, "SELL", "index:", state_ma_a.index[-1], "Profit: £", profit, "Balance:", fullBalance, "Gain:", (fullBalance- originalBalance))
		#print(cnt, "Profit: £", profit,"Balance:",fullBalance, "|BTC:", np.round(btc_balance, 4), "SOLD")
		sell = btc_price_current
		activate_sell = False
		initBalance = fullBalance

	days_since_buy += 1
	previous_degree = degree

	if buy != 0:
		buy_list.loc[index] = int(buy)
	if sell != 0:
		sell_list.loc[index] = int(sell)
		profit_list.loc[index] = int(profit)

def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	return [float('nan') if x == 0 else x for x in values]

print("making graph")
plt.style.use('dark_background')
fig = plt.figure(figsize=(19, 10))
ax1 = fig.add_subplot(111)
fig.tight_layout()

ax1.plot(price_list, "-", color='gray', linewidth=1.5)
ax1.plot(df_ma_charts["mA"], "-", color='#86c16a', linewidth=1.5, label=("buy " + str(ma_a_size)))
ax1.plot(df_ma_charts["mB"], "-", color='#183515', linewidth=4, label=("buy tresh " + str(ma_b_size)))
ax1.plot(df_ma_charts["mC"], "-", color='#bc6454', linewidth=1.5, label=("sell " + str(ma_c_size)))
ax1.plot(df_ma_charts["mD"], "-", color='#511308', linewidth=4, label=("sell tresh " + str(ma_d_size)))
ax1.plot(buy_list["Close"], "o", color='darkgreen', markersize=10)
ax1.plot(sell_list["Close"], "o", color='darkred', markersize=10)

ax1.plot(df_ma_charts["smoothB"].index, df_ma_charts["smoothB"], "-", color='orange', linewidth=4, label=("buy smooth"), alpha=0.5)

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
