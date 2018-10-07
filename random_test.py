import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv

from GameEngine_v010_lastWeek import PlayGame

GE = PlayGame()


games = 1

actions = [0,1,2,3]
#actions = [0,1]

percentChangeLog = pd.DataFrame(columns=["percentChange"])
logCnt = 0

for game in range(games):
    terminal = False
    GE.startGame(evaluation=True)

    while terminal == False:

        action = actions[np.random.choice(np.arange(0, 4), p=[0.25, 0.25, 0.25,0.25])]
        #action = actions[np.random.choice(np.arange(0, 2), p=[0.5, 0.5])]

        new_frame, reward, terminal = GE.nextStep(action)
        #print("action:", action)

    if terminal == True:
        print("game:", logCnt)
        percentChange = GE.getBTCPercentChange()
        percentChangeLog.loc[logCnt] = percentChange
        logCnt += 1
        percentChangeLog.to_csv("/home/andras/PycharmProjects/TradingGame/logs/percentChange.csv", index=True)

        GE.startGame(evaluation=True)
