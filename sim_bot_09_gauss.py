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

# SMA
sma_days_A = 0
sma_days_B = 0
sma_days_C = 0
sma_days_D = 0
std_days_A = 60

# FLASH CRASH TETHER: 59000 - 61000

# START DATE
btc_state_size = 1000 # 33000, 63000,      39000 - 41344
state_start_date_idx = 40000
game_start_date_idx = state_start_date_idx + btc_state_size
game_end_date_idx = 42000

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

def func(x, adj1,adj2):
	return (((x+adj1) ** pw) * adj2).astype(int)

x = [0, 16] # two given datapoints to which the exponential function with power pw should fit
y = [2,4000]
pw = 2
A = np.exp(np.log(y[0]/y[1])/pw)
a = (x[0] - x[1]*A)/(A-1)
b = y[0]/(x[0]+a)**pw
xf = np.linspace(0,16,16)
yCalcualted = func(xf, a, b)

ma_a_size = yCalcualted[0]
ma_b_size = yCalcualted[1]
ma_c_size = yCalcualted[2]
ma_d_size = yCalcualted[3]
ma_e_size = yCalcualted[4]
ma_f_size = yCalcualted[5]
ma_g_size = yCalcualted[6]
ma_h_size = yCalcualted[7]  # 5 days
ma_i_size = yCalcualted[8]
ma_j_size = yCalcualted[9]
ma_k_size = yCalcualted[10]
ma_l_size = yCalcualted[11]  # 85 days
ma_m_size = yCalcualted[12]  # 170 days
ma_n_size = yCalcualted[13]
ma_o_size = yCalcualted[14]
ma_p_size = yCalcualted[15]

std_a_period = std_days_A * 96

df_ma_charts = pd.DataFrame(columns=["mA", "mB", "mC", "mD", "mE", "mF", "mG", "mH", "mI", "mJ", "mK", "mL", "mM", "mN", "mO", "mP", "smoothB"])

price_list = pd.DataFrame(columns=["Close"])

profit_list = pd.DataFrame(columns=["Close"])
buy_list = pd.DataFrame(columns=["Close"])
sell_list = pd.DataFrame(columns=["Close"])

tan_list_A = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_B = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_C = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_D = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_E = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_F = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_G = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_H = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])
tan_list_I = pd.DataFrame(columns=["x", "y", "line", "tan", "degree"])

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
cnt = 0


def calc_tangent(x, y, point_x):
	spline = interpolate.splrep(x, y, k=1)
	line = arange(point_x - 1, point_x + 1)
	point_y = interpolate.splev(point_x, spline, der=0)  # f(point_x)
	fprime = interpolate.splev(point_x, spline, der=1)  # f'(point_x)
	tan = point_y + fprime * (line - point_x)  # tangent
	return point_x, point_y, line, tan, fprime

df_segment = df_BTC.loc[game_start_date_idx:game_end_date_idx]


#for index, row in df_segment.iterrows():
for index in df_segment.index:
	#start_time = timeit.default_timer()
	#print(index, row['Unix'], row['Close'])

	cnt += 1
	days = int(np.round((cnt / 96), 0))
	if cnt % 96 == 0:
		print("DAY", days)

	# State segment for MA calculations
	state_all_ma = df_BTC['Close'].loc[index - 600 - ma_p_size * 2: index]

	# MA Calculations
	state_ma_a = np.round( (talib.DEMA(state_all_ma, timeperiod=ma_a_size)), 6 ).dropna()#.astype('float32')
	state_ma_b = np.round( (talib.DEMA(state_all_ma, timeperiod=ma_b_size)), 6 ).dropna()#.astype('float32')
	state_ma_c = np.round( (talib.DEMA(state_all_ma, timeperiod=ma_c_size)), 6 ).dropna()#.astype('float32')
	state_ma_d = np.round( (talib.DEMA(state_all_ma, timeperiod=ma_d_size)), 6 ).dropna()#.astype('float32')
	state_ma_e = np.round((talib.DEMA(state_all_ma, timeperiod=ma_e_size)), 6).dropna()
	state_ma_f = np.round((talib.DEMA(state_all_ma, timeperiod=ma_f_size)), 6).dropna()
	state_ma_g = np.round((talib.DEMA(state_all_ma, timeperiod=ma_g_size)), 6).dropna()
	state_ma_h = np.round((talib.DEMA(state_all_ma, timeperiod=ma_h_size)), 6).dropna()
	state_ma_i = np.round((talib.DEMA(state_all_ma, timeperiod=ma_i_size)), 6).dropna()
	state_ma_j = np.round((talib.DEMA(state_all_ma, timeperiod=ma_j_size)), 6).dropna()
	state_ma_k = np.round((talib.DEMA(state_all_ma, timeperiod=ma_k_size)), 6).dropna()
	state_ma_l = np.round((talib.DEMA(state_all_ma, timeperiod=ma_l_size)), 6).dropna()
	state_ma_m = np.round((talib.DEMA(state_all_ma, timeperiod=ma_m_size)), 6).dropna()
	state_ma_n = np.round((talib.DEMA(state_all_ma, timeperiod=ma_n_size)), 6).dropna()
	state_ma_o = np.round((talib.DEMA(state_all_ma, timeperiod=ma_o_size)), 6).dropna()
	state_ma_p = np.round((talib.DEMA(state_all_ma, timeperiod=ma_p_size)), 6).dropna()

	# Extra smoothing on top of Moving averages
	state_ma_b_smooth_x = np.asarray(state_ma_b.index)
	state_ma_b_smooth = state_ma_b.rolling(window=2, min_periods=0).mean()
	state_ma_b_smooth = state_ma_b_smooth.astype('float32')

	# Define Current Prices
	btc_price_current = df_BTC['Close'].at[index]
	ma_a_last = state_ma_a.at[index]
	ma_b_last = state_ma_b.at[index]
	ma_c_last = state_ma_c.at[index]
	ma_d_last = state_ma_d.at[index]
	ma_e_last = state_ma_e.at[index]
	ma_f_last = state_ma_f.at[index]
	ma_g_last = state_ma_g.at[index]
	ma_h_last = state_ma_h.at[index]
	ma_i_last = state_ma_i.at[index]
	ma_j_last = state_ma_j.at[index]
	ma_k_last = state_ma_k.at[index]
	ma_l_last = state_ma_l.at[index]
	ma_m_last = state_ma_m.at[index]
	ma_n_last = state_ma_n.at[index]
	ma_o_last = state_ma_o.at[index]
	ma_p_last = state_ma_p.at[index]

	ma_b_smooth_current = state_ma_b_smooth.at[index]

	# Add Current Prices to data frames
	price_list.at[index] = df_BTC['Close'].at[index]
	price_list = price_list.astype('uint16')

	df_ma_charts.at[index] = ma_a_last, ma_b_last, ma_c_last, ma_d_last, ma_e_last, ma_f_last, ma_g_last, ma_h_last, \
							 ma_i_last, ma_j_last, ma_k_last, ma_l_last, ma_m_last, ma_n_last, ma_o_last, ma_p_last, ma_b_smooth_current
	df_ma_charts['mA'] = df_ma_charts['mA'].astype('float32')
	df_ma_charts['mB'] = df_ma_charts['mB'].astype('float32')
	df_ma_charts['mC'] = df_ma_charts['mC'].astype('float32')
	df_ma_charts['mD'] = df_ma_charts['mD'].astype('float32')
	df_ma_charts['smoothB'] = df_ma_charts['smoothB'].astype('float32')

	# 0.004 ------------------------------------------------------------------------------------------------------------

	#  --------------------------------- CHECK FOR INTERSECTIONS ---------------------------------
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

	#  --------------------------------- CHECK IF SLOPE IS INCREASING ---------------------------------
	#print(np.array(state_ma_a.index))

	#point_x, point_y, line, tan, degree = calc_tangent(state_ma_b_smooth_x, state_ma_b_smooth, state_ma_b_smooth_x[-2])
	point_x_a, point_y_a, line_a, tan_a, degree_a = calc_tangent(state_ma_a.index, state_ma_a, state_ma_a.index[-1])
	point_x_b, point_y_b, line_b, tan_b, degree_b = calc_tangent(state_ma_b.index, state_ma_b, state_ma_b.index[-1])
	point_x_c, point_y_c, line_c, tan_c, degree_c = calc_tangent(state_ma_c.index, state_ma_c, state_ma_c.index[-1])
	point_x_d, point_y_d, line_d, tan_d, degree_d = calc_tangent(state_ma_d.index, state_ma_d, state_ma_d.index[-1])
	point_x_e, point_y_e, line_e, tan_e, degree_e = calc_tangent(state_ma_e.index, state_ma_e, state_ma_e.index[-1])
	point_x_f, point_y_f, line_f, tan_f, degree_f = calc_tangent(state_ma_f.index, state_ma_f, state_ma_f.index[-1])


	tan_angle_list_B.at[index, 'Angle'] = np.round(degree_a, 3).astype('float16')

	#
	# if len(tan_angle_list_B) > 20:
	# 	tan_angle_list_B.drop([index-11])
	# 	if tan_angle_list_B.at[index, "Angle"] > tan_angle_list_B.at[(index-10), "Angle"]: # and tan_angle_list_B["Angle"].iloc[-1] >= 0:
	# 		buy_slope_increasing = True
	# 	else:
	# 		buy_slope_increasing = False
	# else:
	# 	buy_slope_increasing = False

	# --------------------------------- VOTING SYSTEM ---------------------------------
	ma_vote_buy_a = 0
	ma_vote_sell_a = 0
	if degree_a > 0.1:
		ma_vote_buy_a = 1
	if degree_a < -0.1:
		ma_vote_sell_a = 1

	ma_vote_buy_b = 0
	ma_vote_sell_b = 0
	if degree_b > 0.1:
		ma_vote_buy_b = 1
	if degree_b < -0.1:
		ma_vote_sell_b = 1

	ma_vote_buy_c = 0
	ma_vote_sell_c = 0
	if degree_c > 0.1:
		ma_vote_buy_c = 1
	if degree_c < -0.1:
		ma_vote_sell_c = 1

	ma_vote_buy_d = 0
	ma_vote_sell_d = 0
	if degree_d > 0.1:
		ma_vote_buy_d = 1
	if degree_d < -0.1:
		ma_vote_sell_d = 1

	ma_vote_buy_e = 0
	ma_vote_sell_e = 0
	if degree_e > 0.1:
		ma_vote_buy_e = 1
	if degree_e < -0.1:
		ma_vote_sell_e = 1

	ma_vote_buy_f = 0
	ma_vote_sell_f = 0
	if degree_f > 0.1:
		ma_vote_buy_f = 1
	if degree_f < -0.1:
		ma_vote_sell_f = 1

	vote_buy_count = ma_vote_buy_a + ma_vote_buy_b + ma_vote_buy_c + ma_vote_buy_d + ma_vote_buy_e + ma_vote_buy_f

	if vote_buy_count >= 5:
		activate_buy = True
	else:
		activate_buy = False

	vote_sell_count = ma_vote_sell_a + ma_vote_sell_b + ma_vote_sell_c + ma_vote_sell_d + ma_vote_sell_e + ma_vote_sell_f
	if vote_sell_count >= 5:
		activate_sell = True
	else:
		activate_sell = False

	# #  ------------------------- BUY CURVE IS CROSSED UPWARDS -------------------------
	# if intersect_ab_cnt > 0:
	# 	intersect_ab_cnt = 0
	# 	# if ma_a_last > ma_b_last:
	# 	# 	tan_list_B.at[index, 'x'] = point_x
	# 	# 	tan_list_B.at[index, 'y'] = point_y
	# 	# 	tan_list_B.at[index, 'line'] = line
	# 	# 	tan_list_B.at[index, 'tan'] = tan
	# 	# 	tan_list_B.at[index, 'degree'] = degree
	# 
	# 	if ma_a_last > ma_b_last and degree > 0.1:
	# 		activate_buy = True
	# 	else:
	# 		activate_buy = False
	# else:
	# 	activate_buy = False
	# 
	# # ------------------------- SELL CURVE IS CROSSED DOWNWARDS -------------------------
	# if intersect_cd_cnt > 0 and btc_balance != 0: #and degree < 0  #and not buy_slope_increasing:
	# 	intersect_cd_cnt = 0
	# 	if ma_c_last <= ma_d_last:
	# 		activate_sell = True
	# 
	# 	if ma_c_last > ma_c_last:
	# 		activate_sell = False
	# else:
	# 	activate_sell = False

	# ------------------------- NEIGHTER HAPPENED -------------------------
	# if not activate_buy and not activate_sell:
	# 	fullBalance = fiat_cash_balance + btc_balance * btc_price_current
	# 	profit = fullBalance - initBalance
	# 	profit = int(np.round(profit, 0))

	# 0.0016 ------------------------------------------------------------------------------------------------------------


	# ------------------------- BUY -------------------------
	buy = 0
	if activate_buy: # and degree >= 0.1:
		try:
			tan_list_A.at[index, 'x'] = point_x_a
			tan_list_A.at[index, 'y'] = point_y_a
			tan_list_A.at[index, 'line'] = line_a
			tan_list_A.at[index, 'tan'] = tan_a
			tan_list_A.at[index, 'degree'] = degree_a

			tan_list_B.at[index, 'x'] = point_x_b
			tan_list_B.at[index, 'y'] = point_y_b
			tan_list_B.at[index, 'line'] = line_b
			tan_list_B.at[index, 'tan'] = tan_b
			tan_list_B.at[index, 'degree'] = degree_b

			tan_list_C.at[index, 'x'] = point_x_c
			tan_list_C.at[index, 'y'] = point_y_c
			tan_list_C.at[index, 'line'] = line_c
			tan_list_C.at[index, 'tan'] = tan_c
			tan_list_C.at[index, 'degree'] = degree_c

			tan_list_D.at[index, 'x'] = point_x_d
			tan_list_D.at[index, 'y'] = point_y_d
			tan_list_D.at[index, 'line'] = line_d
			tan_list_D.at[index, 'tan'] = tan_d
			tan_list_D.at[index, 'degree'] = degree_d

			tan_list_E.at[index, 'x'] = point_x_e
			tan_list_E.at[index, 'y'] = point_y_e
			tan_list_E.at[index, 'line'] = line_e
			tan_list_E.at[index, 'tan'] = tan_e
			tan_list_E.at[index, 'degree'] = degree_e

			tan_list_F.at[index, 'x'] = point_x_f
			tan_list_F.at[index, 'y'] = point_y_f
			tan_list_F.at[index, 'line'] = line_f
			tan_list_F.at[index, 'tan'] = tan_f
			tan_list_F.at[index, 'degree'] = degree_f

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
	sell = 0
	if activate_sell and btc_balance != 0:
		try:
			tan_list_A.at[index, 'x'] = point_x_a
			tan_list_A.at[index, 'y'] = point_y_a
			tan_list_A.at[index, 'line'] = line_a
			tan_list_A.at[index, 'tan'] = tan_a
			tan_list_A.at[index, 'degree'] = degree_a

			tan_list_B.at[index, 'x'] = point_x_b
			tan_list_B.at[index, 'y'] = point_y_b
			tan_list_B.at[index, 'line'] = line_b
			tan_list_B.at[index, 'tan'] = tan_b
			tan_list_B.at[index, 'degree'] = degree_b

			tan_list_C.at[index, 'x'] = point_x_c
			tan_list_C.at[index, 'y'] = point_y_c
			tan_list_C.at[index, 'line'] = line_c
			tan_list_C.at[index, 'tan'] = tan_c
			tan_list_C.at[index, 'degree'] = degree_c

			tan_list_D.at[index, 'x'] = point_x_d
			tan_list_D.at[index, 'y'] = point_y_d
			tan_list_D.at[index, 'line'] = line_d
			tan_list_D.at[index, 'tan'] = tan_d
			tan_list_D.at[index, 'degree'] = degree_d

			tan_list_E.at[index, 'x'] = point_x_e
			tan_list_E.at[index, 'y'] = point_y_e
			tan_list_E.at[index, 'line'] = line_e
			tan_list_E.at[index, 'tan'] = tan_e
			tan_list_E.at[index, 'degree'] = degree_e

			tan_list_F.at[index, 'x'] = point_x_f
			tan_list_F.at[index, 'y'] = point_y_f
			tan_list_F.at[index, 'line'] = line_f
			tan_list_F.at[index, 'tan'] = tan_f
			tan_list_F.at[index, 'degree'] = degree_f
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

	if buy != 0:
		buy_list.at[index] = buy
	if sell != 0:
		sell_list.at[index] = sell
		profit_list.at[index] = profit

	#print(buy_list)
	#print(timeit.default_timer() - start_time)
	#print("\n")

def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	return [float('nan') if x == 0 else x for x in values]

print("making graph")
plt.style.use('dark_background')
fig = plt.figure(figsize=(19, 10))
ax1 = fig.add_subplot(111)
fig.tight_layout()

ax1.plot(price_list, "-", color='gray', linewidth=1)
ax1.plot(df_ma_charts["mA"], "-", color='#ff4040', linewidth=1, label=("MA" + str(ma_a_size)))
ax1.plot(df_ma_charts["mB"], "-", color='#ff7d40', linewidth=1, label=("MA" + str(ma_b_size)))
ax1.plot(df_ma_charts["mC"], "-", color='#ffb440', linewidth=1, label=("MA" + str(ma_c_size)))
ax1.plot(df_ma_charts["mD"], "-", color='#ffeb40', linewidth=1, label=("MA" + str(ma_d_size)))
ax1.plot(df_ma_charts["mE"], "-", color='#85ff40', linewidth=1, label=("MA" + str(ma_e_size)))
ax1.plot(df_ma_charts["mF"], "-", color='#40ff83', linewidth=1, label=("MA" + str(ma_f_size)))

# ax1.plot(df_ma_charts["mG"], "-", color='#40ffd1', linewidth=1, label=("MA" + str(ma_g_size)))
# ax1.plot(df_ma_charts["mH"], "-", color='#40ceff', linewidth=1, label=("MA" + str(ma_h_size)))
# ax1.plot(df_ma_charts["mI"], "-", color='#408bff', linewidth=1, label=("MA" + str(ma_i_size)))
# ax1.plot(df_ma_charts["mJ"], "-", color='#4043ff', linewidth=1, label=("MA" + str(ma_j_size)))
# ax1.plot(df_ma_charts["mK"], "-", color='#8340ff', linewidth=1, label=("MA" + str(ma_k_size)))
# ax1.plot(df_ma_charts["mL"], "-", color='#a840ff', linewidth=1, label=("MA" + str(ma_l_size)))
# ax1.plot(df_ma_charts["mM"], "-", color='#e540ff', linewidth=1, label=("MA" + str(ma_m_size)))
# ax1.plot(df_ma_charts["mN"], "-", color='#ff40c5', linewidth=1, label=("MA" + str(ma_n_size)))
# ax1.plot(df_ma_charts["mO"], "-", color='#ff408e', linewidth=1, label=("MA" + str(ma_o_size)))
# ax1.plot(df_ma_charts["mP"], "-", color='#ff4069', linewidth=1, label=("MA" + str(ma_p_size)))

# ax1.plot(buy_list["Close"], "o", color='darkgreen', markersize=5)
# ax1.plot(sell_list["Close"], "o", color='darkred', markersize=5)



#ax1.plot(df_ma_charts["smoothB"].index, df_ma_charts["smoothB"], "-", color='orange', linewidth=4, label=("buy smooth"), alpha=0.5)



# for i in tan_list_A.index:
# 	if tan_list_A["degree"][i] >= 0:
# 		ax1.plot(tan_list_A["line"][i], tan_list_A["tan"][i], "--r", color='green', linewidth=3)
# 	else:
# 		ax1.plot(tan_list_A["line"][i], tan_list_A["tan"][i], "--r", color='red', linewidth=3)
#
# for i in tan_list_B.index:
# 	if tan_list_B["degree"][i] >= 0:
# 		ax1.plot(tan_list_B["line"][i], tan_list_B["tan"][i], "--r", color='green', linewidth=3)
# 	else:
# 		ax1.plot(tan_list_B["line"][i], tan_list_B["tan"][i], "--r", color='red', linewidth=3)
#
# for i in tan_list_C.index:
# 	if tan_list_C["degree"][i] >= 0:
# 		ax1.plot(tan_list_C["line"][i], tan_list_C["tan"][i], "--r", color='green', linewidth=3)
# 	else:
# 		ax1.plot(tan_list_C["line"][i], tan_list_C["tan"][i], "--r", color='red', linewidth=3)
#
# for i in tan_list_D.index:
# 	if tan_list_D["degree"][i] >= 0:
# 		ax1.plot(tan_list_D["line"][i], tan_list_D["tan"][i], "--r", color='green', linewidth=3)
# 	else:
# 		ax1.plot(tan_list_D["line"][i], tan_list_D["tan"][i], "--r", color='red', linewidth=3)
#
# for i in tan_list_E.index:
# 	if tan_list_E["degree"][i] >= 0:
# 		ax1.plot(tan_list_E["line"][i], tan_list_E["tan"][i], "--r", color='green', linewidth=3)
# 	else:
# 		ax1.plot(tan_list_E["line"][i], tan_list_E["tan"][i], "--r", color='red', linewidth=3)
#
# for i in tan_list_F.index:
# 	if tan_list_F["degree"][i] >= 0:
# 		ax1.plot(tan_list_F["line"][i], tan_list_F["tan"][i], "--r", color='green', linewidth=3)
# 	else:
# 		ax1.plot(tan_list_F["line"][i], tan_list_F["tan"][i], "--r", color='red', linewidth=3)
#
# ax1.plot(tan_list_A["x"], tan_list_A["y"], "o", color='white', markersize=1)
# ax1.plot(tan_list_B["x"], tan_list_B["y"], "o", color='white', markersize=1)
# ax1.plot(tan_list_C["x"], tan_list_C["y"], "o", color='white', markersize=1)
# ax1.plot(tan_list_D["x"], tan_list_D["y"], "o", color='white', markersize=1)
# ax1.plot(tan_list_E["x"], tan_list_E["y"], "o", color='white', markersize=1)
# ax1.plot(tan_list_F["x"], tan_list_F["y"], "o", color='white', markersize=1)






for i in profit_list["Close"].index:
	text = profit_list["Close"][i]

	if int(text) >= 0:
		ax1.annotate(text, xy=(i, sell_list["Close"][i]), size=20, fontweight='bold', color='green')
	else:
		ax1.annotate(text, xy=(i, sell_list["Close"][i]), size=20, fontweight='bold', color='red')

ax1.legend()
plt.show()