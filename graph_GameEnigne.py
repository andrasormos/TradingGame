import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


#results = pd.read_csv("/home/andras/PycharmProjects/keras-RL/GameEngineLog.csv", sep=",", index_col=0)
results = pd.read_csv("/home/andras/PycharmProjects/reinforcement-learning/3-atari/1-breakout/TraderLog.csv", sep=",", index_col=0)

profit = results['profit']

mean_result = np.mean(profit)

print("Mean Profit", mean_result)



fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(results.index, profit, ".", color='r', markersize=1)
plt.title("profit per game")

plt.show()

