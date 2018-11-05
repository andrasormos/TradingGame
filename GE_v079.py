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


sys.setrecursionlimit(10000)
np.set_printoptions(threshold=np.nan, linewidth=300)

class PlayGame(object):
    def __init__(self):
        self.prediction = False

        self.training_df_ETH = pd.read_csv("./cryptoExtract/trainEval/train_BTC_GBP.csv", index_col=0)
        self.eval_df_ETH = pd.read_csv("./cryptoExtract/trainEval/eval_BTC_GBP.csv", index_col=0)

        self.training_df_BTC = pd.read_csv("./cryptoExtract/trainEval/train_ETH_USD.csv", index_col=0)
        self.eval_df_BTC = pd.read_csv("./cryptoExtract/trainEval/eval_ETH_USD.csv", index_col=0)

        print(self.training_df_BTC.Close)
        #
        # self.training_df_ETH.Close = self.training_df_ETH.Close.astype(int)
        # self.training_df_ETH.Volume = self.training_df_ETH.Volume.astype(int)
        # self.eval_df_ETH.Close = self.eval_df_ETH.Close.astype(int)
        # self.eval_df_ETH.Volume = self.eval_df_ETH.Volume.astype(int)
        # self.training_df_BTC.Close = self.training_df_BTC.Close.astype(int)
        # self.training_df_BTC.Volume = self.training_df_BTC.Volume.astype(int)
        # self.eval_df_BTC.Close = self.eval_df_BTC.Close.astype(int)
        # self.eval_df_BTC.Volume = self.eval_df_BTC.Volume.astype(int)

        self.sma = 168
        self.gameLength = 168  # How long the game should go on
        self.timeFrame = 84  # How many data increment should be shown as history. Could be hours, months
        self.amountToSpend = 500  # How much to purchase crypto for
        self.fiatToTrade = 100
        self.smaStartDate = "itsempty"
        self.smaEndDate = "itsempty"
        self.eLogCnt = 0
        self.tLogCnt = 0
        self.oLogCnt = 0
        self.gamesPlayedCnt = 0
        self.df_trainLog = pd.DataFrame(columns=["rewardSum", "profitSum", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "guessUpRightCnt", "guessUpWrongCnt", "guessDownRightCnt", "guessDownWrongCnt", "eps", "frame",])
        self.df_evalLog = pd.DataFrame(columns=["rewardSum", "profitSum", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "guessUpRightCnt", "guessUpWrongCnt", "guessDownRightCnt", "guessDownWrongCnt", "eps", "frame",])
        self.df_evalOverfitLog = pd.DataFrame(columns=["rewardSum", "profitSum", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "guessUpRightCnt", "guessUpWrongCnt", "guessDownRightCnt", "guessDownWrongCnt", "eps", "frame",])
        self.df_actionLog = pd.DataFrame(columns=["BTCPrice", "bought", "sold"])
        self.actionLogOn = False
        self.logNr = ""
        self.evalType = "evalTrain"
        self.epsilon = 1
        self.frameNumber = 0
        self.BTCPercentChange = 0
        self.priceActionMem = pd.DataFrame(columns=["price", "action"])

    def defineLogNr(self, logNr):
        self.logNr = logNr
        self.trainLogPathName = "./logs/trainLog_" + logNr + ".csv"
        self.evalLogPathName = "./logs/evalLog_" + logNr + ".csv"
        self.evalOverfitLogPathName = "./logs/evalOverfitLog_" + logNr + ".csv"
        self.actionLogPathName = "./logs/actionLog_" + logNr + ".csv"

        if os.path.exists(self.trainLogPathName) == True:
            os.remove(self.trainLogPathName)
        if os.path.exists(self.evalLogPathName) == True:
            os.remove(self.evalLogPathName)
        if os.path.exists(self.evalOverfitLogPathName) == True:
            os.remove(self.evalOverfitLogPathName)
        if os.path.exists(self.actionLogPathName) == True:
            os.remove(self.actionLogPathName)

        self.df_trainLog.to_csv(self.trainLogPathName, mode='a', header=True)
        self.df_evalLog.to_csv(self.evalLogPathName, mode='a', header=True)
        self.df_evalOverfitLog.to_csv(self.evalOverfitLogPathName, mode='a', header=True)
        self.df_actionLog.to_csv(self.actionLogPathName, mode='a', header=True)

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setFrameNumber(self, frameNumber):
        self.frameNumber = frameNumber

    def setEvalType(self, evalType):
        self.evalType = evalType

    def startGame(self):
        self.aLogCnt = 0
        self.initialBalance = 10000  # Starting Money
        self.cashBalance = self.initialBalance
        self.BTC_Balance = 0  # BTC to start with

        if self.evalType == "evalReal":
            self.df_BTC = self.eval_df_BTC
            self.df_ETH = self.eval_df_ETH
        elif self.evalType == "evalTrain":
            self.df_BTC = self.training_df_BTC
            self.df_ETH = self.training_df_ETH
        elif self.evalType == "evalOverfit":
            self.df_BTC = self.training_df_BTC
            self.df_ETH = self.training_df_ETH

        #print(self.df_BTC.Close)

        self.dataSize = len(self.df_BTC.index)
        self.startIndex, self.endIndex, self.smaStartIndex, self.smaEndIndex = self.randomChart()

        if self.evalType == "evalReal":
            self.df_segment_BTC = self.df_BTC.loc[self.endIndex: self.startIndex]
            self.df_segment_BTC_SMA = self.df_BTC.loc[self.smaEndIndex: self.smaStartIndex]
            self.df_segment_ETH = self.df_ETH.loc[self.endIndex: self.startIndex]
            self.df_segment_ETH_SMA = self.df_ETH.loc[self.smaEndIndex: self.smaStartIndex]

        elif self.evalType == "evalTrain" or "evalOverfit":
            self.df_segment_BTC = self.df_BTC.loc[self.endIndex: self.startIndex]
            self.df_segment_BTC_SMA = self.df_BTC.loc[self.smaEndIndex: self.smaStartIndex]
            self.df_segment_ETH = self.df_ETH.loc[self.endIndex: self.startIndex]
            self.df_segment_ETH_SMA = self.df_ETH.loc[self.smaEndIndex: self.smaStartIndex]


        #print(self.df_segment_BTC.Close)

        self.sumProfit = 0
        self.guessUpCnt = 0
        self.guessDownCnt = 0
        self.guessSkipCnt = 0
        self.guessUpRightCnt = 0
        self.guessUpWrongCnt = 0
        self.guessDownRightCnt = 0
        self.guessDownWrongCnt = 0
        self.guessCnt = 0
        self.percentProfitReward = 0
        self.rewardList = []
        self.rewardSum = 0
        self.profit = 0
        self.fiatMoneyInvested = 0
        self.btcFuture = 0
        self.btcPresent = 0
        self.fullBalance = self.cashBalance
        self.prevFullBalance = self.fullBalance
        self.done = False
        self.cnt = 0
        self.profit = 0

    def randomChart(self):
        startIndex = randint((self.timeFrame + self.gameLength), (self.dataSize - 2 - self.sma))
        endIndex = startIndex - self.timeFrame + 1

        smaStartIndex = startIndex + self.sma
        smaEndIndex = endIndex

        return startIndex, endIndex, smaStartIndex, smaEndIndex

    def nextStep(self, action):
        hours = 24
        if self.cnt == 0:
            for i in range(hours):
                self.priceActionMem.loc[i] = 0, 0
        self.cnt = self.cnt + 1
        
        # --------------------------- PRESENT TIME - JUDGE PAST ACTION  ----------------------------
        reward = 0
        if self.cnt > hours:
            btcT0 = self.priceActionMem.price[0]
            btcT24 = self.priceActionMem.price[hours - 1]
            actionT24 = self.priceActionMem.action[hours - 1]

            if btcT0 >= btcT24 and actionT24 == 1:
                reward = 1
                self.guessUpRightCnt += 1
            if btcT0 >= btcT24 and actionT24 == 2:
                reward = -1
                self.guessDownWrongCnt += 1
            if btcT0 >= btcT24 and actionT24 == 3:
                reward = 0

            if btcT0 < btcT24 and actionT24 == 1:
                reward = -1
                self.guessUpWrongCnt += 1
            if btcT0 < btcT24 and actionT24 == 2:
                reward = 1
                self.guessDownRightCnt += 1
            if btcT0 < btcT24 and actionT24 == 3:
                reward = 0
        else:
            reward = 0

        # --------------------------- PROGRESS TO FUTURE  ----------------------------
        # --------------------------- ADD NEW ROW DATA AND FORGET PREVIOUS ----------------------------
        self.endIndex = self.endIndex - 1
        self.nextRow_BTC = self.df_BTC.loc[[self.endIndex]]

        self.df_segment_BTC = pd.concat([self.nextRow_BTC, self.df_segment_BTC])
        self.df_segment_BTC = self.df_segment_BTC.drop(self.df_segment_BTC.index[len(self.df_segment_BTC) - 1])
        # print("DF SEGMENT")
        # print(self.df_segment_BTC)

        self.nextRow_ETH = self.df_ETH.loc[[self.endIndex]]
        self.df_segment_ETH = pd.concat([self.nextRow_ETH, self.df_segment_ETH])
        self.df_segment_ETH = self.df_segment_ETH.drop(self.df_segment_ETH.index[len(self.df_segment_ETH) - 1])

        actionT0 = action
        btcT0 = self.nextRow_BTC["Close"][self.endIndex]
        self.priceActionMem.price = self.priceActionMem.price.shift(1)
        self.priceActionMem.action = self.priceActionMem.action.shift(1)
        self.priceActionMem.loc[0] = btcT0, actionT0

        # -------------------------------------- GAME ENDS IF THESE ARE MET -------------------------------------------
        if self.cnt == self.gameLength:
            self.done = True

        if self.done == True:
            self.gamesPlayedCnt += 1

        # --------------------------- LOG DATA ----------------------------
        self.rewardSum += reward
        self.profit = 0

        if action == 1:
            self.guessUpCnt += 1
        if action == 2:
            self.guessDownCnt += 1
        if action == 3 or action == 0:
            self.guessSkipCnt += 1

        # # --------------------- ACTION LOG ----------------------
        # if self.actionLogOn == True:
        #     if self.actionReceived == 1:
        #         bought = self.btcPresent
        #         sold = 0
        #     if self.actionReceived == 2:
        #         bought = 0
        #         sold = self.btcPresent
        #     if self.actionReceived == 3:
        #         bought = 0
        #         sold = 0
        #     self.aLogCnt += 1
        #     self.df_actionLog.loc[self.aLogCnt] = self.btcPresent, bought, sold
        #     self.df_actionLog.to_csv(self.actionLogPathName, index=True)

        # ------------------------------------------  WRITE EVALUATION LOG  ------------------------------------------
        if self.done == True:
            if self.evalType == "evalReal":
                df = pd.DataFrame(columns=["rewardSum", "profitSum", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "guessUpRightCnt", "guessUpWrongCnt", "guessDownRightCnt", "guessDownWrongCnt", "eps", "frame",])
                df.loc[self.eLogCnt] = self.rewardSum, self.profit, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt, self.guessUpRightCnt, self.guessUpWrongCnt, self.guessDownRightCnt, self.guessDownWrongCnt,  self.epsilon, self.frameNumber
                df.to_csv(self.evalLogPathName, mode='a', header=False)
                self.eLogCnt += 1

            if self.evalType == "evalOverfit":
                df = pd.DataFrame(columns=["rewardSum", "profitSum", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "guessUpRightCnt", "guessUpWrongCnt", "guessDownRightCnt", "guessDownWrongCnt", "eps", "frame",])
                df.loc[self.oLogCnt] = self.rewardSum, self.profit, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt, self.guessUpRightCnt, self.guessUpWrongCnt, self.guessDownRightCnt, self.guessDownWrongCnt,  self.epsilon, self.frameNumber
                df.to_csv(self.evalOverfitLogPathName, mode='a', header=False)
                self.oLogCnt += 1

            if self.evalType == "evalTrain":
                df = pd.DataFrame(columns=["rewardSum", "profitSum", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "guessUpRightCnt", "guessUpWrongCnt", "guessDownRightCnt", "guessDownWrongCnt", "eps", "frame",])
                df.loc[self.tLogCnt] = self.rewardSum, self.profit, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt, self.guessUpRightCnt, self.guessUpWrongCnt, self.guessDownRightCnt, self.guessDownWrongCnt,  self.epsilon, self.frameNumber
                df.to_csv(self.trainLogPathName, mode='a', header=False)
                self.tLogCnt += 1
        image = self.getChartImage(self.timeFrame)

        return image, reward, self.done

    def getCurrentData(self):
        return self.df_segment_BTC["Close"]

    def getBTCPercentChange(self):
        return self.BTCPercentChange

    def getActionTaken(self):
        return self.actionReceived

    def getProfit(self):
        return (self.fullBalance - self.initialBalance)

    def getChartData(self):
        image = self.getChartImage(self.timeFrame)
        return image

    def getCash(self):
        return self.cashBalance

    def getBTC(self):
        return self.BTC_Balance

    def getChartImage(self, timeFrame):
        def scale_list(x, to_min, to_max):
            def scale_number(unscaled, to_min, to_max, from_min, from_max):
                return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min
            if len(set(x)) == 1:
                #print("SET(X) == 1")
                return [np.floor((to_max + to_min) / 2)] * len(x)
            else:
                return [scale_number(i, to_min, to_max, min(x), max(x)) for i in x]

        timeFrame = timeFrame
        PRICE_RANGE = timeFrame
        half_scale_size = int(PRICE_RANGE / 2)
        #half_scale_size = int(PRICE_RANGE)

        # -----------------  BTC  -----------------
        closes_BTC = self.df_segment_BTC["Close"]
        closes_BTC = closes_BTC[::-1]
        close_data_together_BTC = list(np.round(scale_list(closes_BTC[timeFrame - timeFrame: timeFrame], 0, half_scale_size - 1), 0))
        graph_close_BTC = close_data_together_BTC[0:PRICE_RANGE]

        # -----------------  ETH  -----------------
        closes_ETH = self.df_segment_ETH["Close"]
        closes_ETH = closes_ETH[::-1]
        close_data_together_ETH = list(np.round(scale_list(closes_ETH[timeFrame - timeFrame: timeFrame], 0, half_scale_size - 1), 0))
        graph_close_ETH = close_data_together_ETH[0:PRICE_RANGE]

        # -----------------  BTC SMA  -----------------

        btcSMA = self.df_segment_BTC_SMA["Close"].rolling(self.sma).mean()


        btcSMA = btcSMA[::-1]
        btcSMA = list(np.round(scale_list(btcSMA[timeFrame - timeFrame: timeFrame], 0, half_scale_size - 1), 0))
        btcSMA = btcSMA[0:PRICE_RANGE]


        # -----------------  ETH SMA  -----------------
        ethSMA = self.df_segment_ETH_SMA["Close"].rolling(self.sma).mean()
        ethSMA = ethSMA[::-1]
        ethSMA = list(np.round(scale_list(ethSMA[timeFrame - timeFrame: timeFrame], 0, half_scale_size - 1), 0))
        ethSMA = ethSMA[0:PRICE_RANGE]


        def graphRender(data):
            blank_matrix_close = np.zeros(shape=(half_scale_size, timeFrame))
            x_ind = 0
            previous_pixel = 0

            for next_pixel in data:
                blank_matrix_close[int(next_pixel), x_ind] = 150
                plus = True
                if x_ind == 0:
                    previous_pixel = next_pixel

                difference = int((previous_pixel - next_pixel))
                absDifference = abs(difference)
                previous_pixel = next_pixel

                for diff in range(absDifference):

                    if difference >= 0:
                        blank_matrix_close[int(next_pixel), x_ind] = 100
                        next_pixel = (next_pixel + 1).astype(np.uint8)
                        blank_matrix_close[next_pixel, x_ind] = 100
                    if difference < 0:
                        blank_matrix_close[int(next_pixel), x_ind] = 200
                        next_pixel = (next_pixel - 1).astype(np.uint8)
                        blank_matrix_close[next_pixel, x_ind] = 160

                x_ind += 1
            blank_matrix_close = blank_matrix_close[::-1]
            return blank_matrix_close


        # -----------------  CREATE CHARTS  -----------------
        BTC = graphRender(graph_close_BTC)
        btcSMA = graphRender(btcSMA)

        ETH = graphRender(graph_close_ETH)
        ethSMA = graphRender(ethSMA)

        #-----------------  OVERLAY THEM  -----------------
        c = 255 - BTC
        btcSMA = np.asarray(btcSMA)
        BTC = np.asarray(BTC)
        np.putmask(btcSMA, c < btcSMA, c)
        BTC += btcSMA

        c = 255 - ETH
        ethSMA = np.asarray(ethSMA)
        ETH = np.asarray(ETH)
        np.putmask(ethSMA, c < ethSMA, c)
        ETH += ethSMA

        #stackedCharts = SMA
        stackedCharts = np.vstack([BTC, ETH])

        def reMap(OldValue,OldMin,OldMax,NewMin,NewMax,MinLimit,MaxLimit):

            rescaled = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            if rescaled > MinLimit:
                rescaled = MinLimit

            if rescaled < MaxLimit:
                rescaled = MaxLimit
            return rescaled

        def profitMeter(chart):
            matrix = np.zeros(shape=(half_scale_size * 2, timeFrame))
            yAxis = reMap(self.BTCPercentChange, -1.5,1.5, self.timeFrame,0, self.timeFrame, 0)
            xAxis = self.timeFrame / 2
            radius = reMap(self.fiatMoneyInvested, 0,2000,3,10, 10, 1)

            if self.fiatMoneyInvested < 1:
                radius = 1

            rr, cc = draw.circle(yAxis, xAxis, radius=2, shape=matrix.shape)
            matrix[rr, cc] = 100

            c = 255 - matrix
            np.putmask(chart, c < chart, c)
            chart += matrix

            return chart

        #stackedCharts = profitMeter(stackedCharts)

        return stackedCharts


df = pd.DataFrame(columns=['profit'])
cnt = 0
terminal = False
restart = False


def HourLater(action):
    global restart
    global terminal
    global cnt

    plt.close()

    if restart == True:
        restart = False
        plt.style.use('seaborn')
        df_segment_BTC = test.getChartData()
        plt.imshow(df_segment_BTC, cmap='gray')
        # data = test.getCurrentData()
        # plot(data, "-", color='g', linewidth=1)

        buyCom = plt.axes([0.9, 0.2, 0.1, 0.075])
        buyButt = Button(buyCom, 'UP', color='red', hovercolor='green')
        buyButt.on_clicked(_buy)

        sellCom = plt.axes([0.9, 0.1, 0.1, 0.075])
        sellButt = Button(sellCom, 'DOWN', color='red', hovercolor='green')
        sellButt.on_clicked(_sell)

        skipCom = plt.axes([0.9, 0.0, 0.1, 0.075])
        skipButt = Button(skipCom, 'SKIP', color='red', hovercolor='green')
        skipButt.on_clicked(_skip)

        dollMeter = plt.axes([0.9, 0.7, 0.1, 0.075])
        dollText = TextBox(dollMeter, 'Dollar', color='grey', initial=test.getCash())

        btcMeter = plt.axes([0.9, 0.6, 0.1, 0.075])
        btcMeter = TextBox(btcMeter, 'BTC', color='grey', initial=test.getBTC())

        profitMeter = plt.axes([0.9, 0.5, 0.1, 0.075])
        profitMeter = TextBox(profitMeter, 'Profit', color='grey', initial=test.getProfit())

        plt.show()

    if terminal == True:
        plt.style.use('dark_background')
        df.loc[cnt] = test.profit
        cnt += 1
        df.to_csv("Human_Trader_Log.csv", index=True)

        df_segment_BTC = test.getChartData()
        plt.imshow(df_segment_BTC, cmap='gray')

        # data = test.getCurrentData()
        # plot(data, "-", color='g', linewidth=1)

        dollMeter = plt.axes([0.9, 0.7, 0.1, 0.075])
        dollText = TextBox(dollMeter, 'Dollar', color='grey', initial=test.getCash())

        btcMeter = plt.axes([0.9, 0.6, 0.1, 0.075])
        btcMeter = TextBox(btcMeter, 'BTC', color='grey', initial=test.getBTC())

        profitMeter = plt.axes([0.9, 0.5, 0.1, 0.075])
        profitMeter = TextBox(profitMeter, 'Profit', color='grey', initial=test.getProfit())

        endMeter = plt.axes([0.9, 0.4, 0.1, 0.075])
        if test.getProfit() < 0:
            endMeter = TextBox(endMeter, '', color='red', initial="LOST!")
        else:
            endMeter = TextBox(endMeter, '', color='green', initial="WON!")

        plt.show()

        profit = 0
        terminal = False
        restart = True
        test.startGame()
        HourLater(1)

    else:
        chart, r_t, terminal = test.nextStep(action)

        df_segment_BTC = test.getChartData()
        plt.imshow(chart, cmap='gray')

        buyCom = plt.axes([0.9, 0.2, 0.1, 0.075])
        buyButt = Button(buyCom, 'UP', color='red', hovercolor='green')
        buyButt.on_clicked(_buy)

        sellCom = plt.axes([0.9, 0.1, 0.1, 0.075])
        sellButt = Button(sellCom, 'DOWN', color='red', hovercolor='green')
        sellButt.on_clicked(_sell)

        skipCom = plt.axes([0.9, 0.0, 0.1, 0.075])
        skipButt = Button(skipCom, 'SKIP', color='red', hovercolor='green')
        skipButt.on_clicked(_skip)

        dollMeter = plt.axes([0.9, 0.7, 0.1, 0.075])
        dollText = TextBox(dollMeter, 'Dollar', color='grey', initial=test.getCash())

        btcMeter = plt.axes([0.9, 0.6, 0.1, 0.075])
        btcMeter = TextBox(btcMeter, 'BTC', color='grey', initial=test.getBTC())

        profitMeter = plt.axes([0.9, 0.5, 0.1, 0.075])
        profitMeter = TextBox(profitMeter, 'Profit', color='grey', initial=test.getProfit())

        plt.show()

def printBalances():
    print("Dollar = $", test.getCash(), sep='', end='')
    print("\n", sep='', end='')
    print("BTC =    ", test.getBTC(), " BTC", sep='', end='')
    print("\n", sep='', end='')
    print("PROFIT = ", test.getProfit(), sep='', end='')
    print("\n")

def _buy(event):
    global terminal
    HourLater(1)

def _sell(event):
    global terminal
    HourLater(2)

def _skip(event):
    global terminal
    HourLater(3)

def newGame():
    test.startGame()
    df_segment_BTC = test.getChartData()
    plt.imshow(df_segment_BTC, cmap='gray')

    # data = test.getCurrentData()
    # plot(data, "-", color='g', linewidth=1)

    buyCom = plt.axes([0.9, 0.2, 0.1, 0.075])
    buyButt = Button(buyCom, 'UP', color='red', hovercolor='green')
    buyButt.on_clicked(_buy)

    sellCom = plt.axes([0.9, 0.1, 0.1, 0.075])
    sellButt = Button(sellCom, 'DOWN', color='red', hovercolor='green')
    sellButt.on_clicked(_sell)

    skipCom = plt.axes([0.9, 0.0, 0.1, 0.075])
    skipButt = Button(skipCom, 'SKIP', color='red', hovercolor='green')
    skipButt.on_clicked(_skip)

    dollMeter = plt.axes([0.9, 0.7, 0.1, 0.075])
    dollText = TextBox(dollMeter, 'Dollar', color='grey', initial=test.getCash())

    btcMeter = plt.axes([0.9, 0.6, 0.1, 0.075])
    btcMeter = TextBox(btcMeter, 'BTC', color='grey', initial=test.getBTC())

    profitMeter = plt.axes([0.9, 0.5, 0.1, 0.075])
    profitMeter = TextBox(profitMeter, 'Profit', color='grey', initial=test.getProfit())

    plt.show()


if __name__ == "__main__":
    test = PlayGame()
    test.defineLogNr("00")
    newGame()



