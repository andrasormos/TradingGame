import pandas as pd
import matplotlib.pyplot as plt

from game_engines.game_versions.GameEngine_v010_lastWeek import PlayGame

GE = PlayGame()


games = 1000

actions = [0,1,2,3]
#actions = [0,1]

percentChangeLog = pd.DataFrame(columns=["percentChange"])
logCnt = 0

if 1 == 2:
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


percentChange = pd.read_csv("/home/andras/PycharmProjects/TradingGame/logs/percentChange.csv", sep=",", index_col=0)

fig = plt.figure(figsize=(12, 10))



# AX 1
ax1 = fig.add_subplot(211)
ax1.plot(percentChange, "-", color='b', linewidth=1)
#ax1.set_ylim([priceMin, priceMax])


plt.show()

