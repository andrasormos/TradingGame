import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#df.columns = ["update", "reward_mean", "loss"]

works = pd.read_csv("/home/andras/PycharmProjects/TradingGame/scorePerGameLog.csv", sep=",", index_col=0)
works.loss = works.loss.str.extract(r'.*?([0.0-9.0]+).*?', expand=True)
works.loss = works.loss.astype(float)



fig = plt.figure()
plt.title("Score per game")
ax1 = fig.add_subplot(111)
#ax1.set_ylim([0,15])
#ax1.set_xlim([0,15])
ax1.plot(works["update"], works["reward_mean"],".", color='b', markersize=0.6)




plt.show()