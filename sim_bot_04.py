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

# SIMULATES THE ACTUAL PASSAGE OF TIME IN 15 MINUTES INCREMENTS


def smoothListGaussian(list, strippedXs=False, degree=5):
	window = degree * 2 - 1
	weight = np.array([1.0] * window)
	weightGauss = []
	for i in range(window):
		i = i - degree + 1
		frac = i / float(window)
		gauss = 1 / (np.exp((4 * (frac)) ** 2))
		weightGauss.append(gauss)
	weight = np.array(weightGauss) * weight
	smoothed = [0.0] * (len(list) - window)
	for i in range(len(smoothed)):
		smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)
	return smoothed

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
btc_state_size = 1000 # 33000, 63000,      40000 - 41344
start_date = 39000
end_date = start_date + btc_state_size
game_end_date = 40344

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

smooth_list_B = []
std_list_A = []
price_list = pd.DataFrame(columns=["Close"])
profit_list = pd.DataFrame(columns=["Price"])
buy_list = pd.DataFrame(columns=["Price"])
sell_list = pd.DataFrame(columns=["Price"])

point_x_list_B = []
point_y_list_B = []
line_x_list_B = []
tan_x_list_B = []

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
	price_list.loc[btc_state.index[-1]] = btc_state.iloc[-1]

	# MA A
	btc_state_ma_a = pd.concat([btc_state_ma_a, next_state_row])
	btc_state_ma_a = btc_state_ma_a.drop(btc_state_ma_a.index[0])
	btc_state_ma_a_applied = talib.DEMA(btc_state_ma_a["Close"], timeperiod=ma_a)
	#btc_state_ma_a_applied = btc_state_ma_a_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_a_price_current = btc_state_ma_a_applied.iloc[-1]
	sma_list_A.loc[btc_state_ma_a_applied.index[-1]] = btc_state_ma_a_applied.iloc[-1]


	# MA B
	btc_state_ma_b = pd.concat([btc_state_ma_b, next_state_row])
	btc_state_ma_b = btc_state_ma_b.drop(btc_state_ma_b.index[0])
	btc_state_ma_b_applied = talib.DEMA(btc_state_ma_b["Close"], timeperiod=ma_b)
	#btc_state_ma_b_applied = btc_state_ma_b_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_b_price_current = btc_state_ma_b_applied.iloc[-1]
	sma_list_B.loc[btc_state_ma_b_applied.index[-1]] = btc_state_ma_b_applied.iloc[-1]

	# MA C
	btc_state_ma_c = pd.concat([btc_state_ma_c, next_state_row])
	btc_state_ma_c = btc_state_ma_c.drop(btc_state_ma_c.index[0])
	btc_state_ma_c_applied = talib.DEMA(btc_state_ma_c["Close"], timeperiod=ma_c)
	#btc_state_ma_c_applied = btc_state_ma_c_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_c_price_current = btc_state_ma_c_applied.iloc[-1]
	sma_list_C.loc[btc_state_ma_c_applied.index[-1]] = btc_state_ma_c_applied.iloc[-1]

	# TEST MA
	btc_state_ma_d = pd.concat([btc_state_ma_d, next_state_row])
	btc_state_ma_d = btc_state_ma_d.drop(btc_state_ma_d.index[0])
	btc_state_ma_d_applied = talib.DEMA(btc_state_ma_d["Close"], timeperiod=ma_d)
	#btc_state_ma_d_applied = btc_state_ma_d_applied.loc[btc_state.index[0]:]  # crop the beginning using btc state's range
	btc_state_ma_d_price_current = btc_state_ma_d_applied.iloc[-1]
	sma_list_D.loc[btc_state_ma_d_applied.index[-1]] = btc_state_ma_d_applied.iloc[-1]

	# GAUSSIAN MA B
	list_y = np.asarray(btc_state_ma_b_applied[-1004:])
	list_x = np.asarray(btc_state_ma_b_applied.index[-1004:])
	first_index = list_x[0]
	list_y_gauss = smoothListGaussian(list_y, degree=8)
	list_x_gauss = np.arange((first_index + 8), (first_index + 8 + len(list_y_gauss)) )
	plot_x_gauss = np.arange((cnt-1004 + 8), (cnt-1004 + 8 + len(list_y_gauss)) )
	list_y_gauss_current = list_y_gauss[-1]

	def draw_tangent(x, y, point_x):
		# interpolate the data with point_x spline
		spline = interpolate.splrep(x, y)
		line = arange(point_x - 50, point_x + 50)
		point_y = interpolate.splev(point_x, spline, der=0)  # f(point_x)
		fprime = interpolate.splev(point_x, spline, der=1)  # f'(point_x)
		tan = point_y + fprime * (line - point_x)  # tangent

		ax1.plot(point_x, point_y, 'o')
		ax1.plot(line, tan, '--r')  # '--r'

		pi = 22 / 7
		radian = fprime
		degree = radian * (180 / pi)
		#print(degree)

		return point_x, point_y, line, tan, degree


	#smooth_list_B.append(list_y_gauss_current)

	price_index = np.asarray(btc_state_ma_b_applied[-1004:])
	t = np.asarray(btc_state_ma_b_applied.index[-1004:])

	if 1==2:
		fig = plt.figure(figsize=(12, 10))
		ax1 = fig.add_subplot(111)
		ax1.plot(btc_state["Close"], "-", color='black', linewidth=2)
		ax1.plot(btc_state_ma_a_applied, "-", color='blue', linewidth=2, label=("sma" + str(sma_days_A)))
		ax1.plot(btc_state_ma_b_applied, "-", color='darkgreen', linewidth=2, label=("ema" + str(sma_days_B)))
		ax1.plot(btc_state_ma_c_applied, "-", color='darkred', linewidth=2, label=("sma" + str(sma_days_C)))
		ax1.plot(list_x_gauss, list_y_gauss, "-", color='orange', linewidth=2, label=("buy smooth"))

		draw_tangent(list_x_gauss, list_y_gauss, list_x_gauss[-1])

		#draw_tangent(t, bspl_y, 40500)

		ax1.legend()
		plt.show()

	buy = 0
	sell = 0
	if len(sma_list_A) > 2:
		interPrice = np.asarray(price_list)
		interA = np.asarray(sma_list_A["Close"].values)
		# print(interA)
		# print("\n")
		interB = np.asarray(sma_list_B["Close"].values)
		interC = np.asarray(sma_list_C["Close"].values)
		interD = np.asarray(sma_list_D["Close"].values)
		intersect_ab_cnt = len(np.argwhere(np.diff(np.sign(interB - interA))).flatten())

		intersect_ac_cnt = len(np.argwhere(np.diff(np.sign(interC - interA))).flatten())
		intersect_cd_cnt = len(np.argwhere(np.diff(np.sign(interC - interD))).flatten())
		intersect_pc_cnt = len(np.argwhere(np.diff(np.sign(interPrice - interC))).flatten())

	# CHECK SLOPE
	# print(btc_state_ma_b["Close"])


	ma_b_list = np.asarray(btc_state_ma_b_applied.values)

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
		if btc_state_ma_a_price_current > btc_state_ma_b_price_current: # and buy_slope_increasing:

			activate_buy = True
		else:
			activate_buy = False
		past_intersect_ab_cnt = intersect_ab_cnt

	# SELL CURVE IS CROSSED DOWNWARDS
	if intersect_cd_cnt > past_intersect_cd_cnt and not buy_slope_increasing and btc_balance != 0:
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
		print(cnt, "BUY ", )
		#print(df_BTC["Date"][i], "Profit: £", profit, "Balance:",fullBalance, "|BTC:", np.round(btc_balance, 4), "BOUGHT")
		buy = btc_price_current
		activate_buy = False

		# point_x, point_y, line, tan, fprime = draw_tangent(plot_x_gauss, list_y_gauss, plot_x_gauss[-1])
		# print(cnt, "fprime", fprime)

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

		# point_x, point_y, line, tan, fprime = draw_tangent(plot_x_gauss, list_y_gauss, plot_x_gauss[-1])
		# print(cnt, "fprime", fprime)

	days_since_buy += 1

	if buy != 0:
		buy_list.loc[btc_state.index[-1]] = int(buy)

	if sell != 0:
		sell_list.loc[btc_state.index[-1]] = int(sell)
		profit_list.loc[btc_state.index[-1]] = int(profit)


def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	return [float('nan') if x == 0 else x for x in values]


fig = plt.figure(figsize=(19, 10))
ax1 = fig.add_subplot(111)
fig.tight_layout()

ax1.plot(price_list, "-", color='gray', linewidth=1.5)
ax1.plot(sma_list_A, "-", color='lightgreen', linewidth=2, label=("buy " + str(ma_a)))
ax1.plot(sma_list_B, "-", color='darkgreen', linewidth=2, label=("buy tresh " + str(ma_b)))
ax1.plot(sma_list_C, "-", color='pink', linewidth=2, label=("sell " + str(ma_c)))
ax1.plot(sma_list_D, "-", color='darkred', linewidth=2, label=("sell tresh " + str(ma_d)))

# list_x_gauss = np.arange((-8), (len(smooth_list_B)-8) )
# ax1.plot(list_x_gauss, smooth_list_B, "-", color='orange', linewidth=2, label=("buy smooth"))


ax1.plot(buy_list["Price"], "o", color='darkgreen', markersize=10)
ax1.plot(sell_list["Price"], "o", color='darkred', markersize=10)

print("sell_list")
print(sell_list["Price"].index)
print("\n")

print("profit list")
print(profit_list["Price"].index)
print("\n")

# xi = [i for i in range(0, len(sell_list["Price"]))]

for i in profit_list["Price"].index:
	text = profit_list["Price"][i]

	if int(text) >= 0:
		ax1.annotate(text, xy=(i, sell_list["Price"][i]), size=20, fontweight='bold', color='green')
	else:
		ax1.annotate(text, xy=(i, sell_list["Price"][i]), size=20, fontweight='bold', color='red')


ax1.legend()

plt.show()