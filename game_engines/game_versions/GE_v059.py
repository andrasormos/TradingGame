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


        dateParse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
        self.training_df_BTC = pd.read_csv("./new_crypto/Gdax_ETHUSD_1h_close_train.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)
        self.eval_df_BTC = pd.read_csv("./new_crypto/Gdax_ETHUSD_1h_close_eval.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)

        self.training_df_ETH = pd.read_csv("./new_crypto/Gdax_BTCUSD_1h_close_train.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)
        self.eval_df_ETH = pd.read_csv("./new_crypto/Gdax_BTCUSD_1h_close_eval.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)

        self.eLogCnt = 0
        self.tLogCnt = 0
        self.oLogCnt = 0
        self.gamesPlayedCnt = 0

        self.df_trainLog = pd.DataFrame(columns=["sumPercent", "rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "eps", "frame"])
        self.df_evalLog = pd.DataFrame(columns=["sumPercent", "rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "eps", "frame"])
        self.df_evalOverfitLog = pd.DataFrame(columns=["sumPercent", "rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "eps", "frame"])

        self.actionLogOn = True
        self.df_actionLog = pd.DataFrame(columns=["BTCPrice", "bought", "sold"])
        self.logNr = ""
        self.evalType = "evalTrain"
        self.epsilon = 1
        self.frameNumber = 0

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

        if self.evalType == "evalReal":
            #print("evalreal")
            self.df_BTC = self.eval_df_BTC
            self.df_ETH = self.eval_df_ETH
        elif self.evalType == "evalTrain":
            #print("evalTrain")
            self.df_BTC = self.training_df_BTC
            self.df_ETH = self.training_df_ETH
        elif self.evalType == "evalOverfit":
            #print("evalOverfit")
            self.df_BTC = self.training_df_BTC
            self.df_ETH = self.training_df_ETH
        else:
            print("incorrect eval df name")

        self.gameLength = 168  # How long the game should go on
        self.timeFrame = 84  # How many data increment should be shown as history. Could be hours, months
        self.timeStepSize = "H"  # Does nothing atm
        self.amountToSpend = 500  # How much to purchase crypto for
        self.initialBalance = 10000  # Starting Money
        self.poundsToTrade = 100
        self.cashBalance = self.initialBalance
        self.BTC_Balance = 0  # BTC to start with
        self.actionTaken = 0
        self.sumProfit = 0

        self.terminal = False
        self.guessedRightCnt = 0
        self.guessedWrongCnt = 0

        self.guessUpCnt = 0
        self.guessDownCnt = 0
        self.guessSkipCnt = 0
        self.guessCnt = 0

        self.percentProfitReward = 0
        self.rewardList = []
        self.guessOutcome = 0
        self.sumRightPerc = 0
        self.sumWrongPerc = 0
        self.sumPercent = 0
        self.rewardSum = 0
        self.profitSum = 0
        self.realMoneySpent = 0

        self.priceAtTimeOfPurchase = 0

        self.dataSize = len(self.df_BTC.index)
        self.startDate, self.endDate, self.startIndex, self.endIndex = self.randomChart()

        if self.evalType == "evalReal":
            self.df_segment_BTC = self.df_BTC.loc[self.endDate: self.startDate]
            self.df_segment_ETH = self.df_ETH.loc[self.endDate: self.startDate]
            #print("EVAL DF")
            #print(self.df_segment_BTC)
        elif self.evalType == "evalTrain" or "evalOverfit":
            self.df_segment_BTC = self.df_BTC.loc[self.startDate: self.endDate]
            self.df_segment_ETH = self.df_ETH.loc[self.startDate: self.endDate]
            #print("TRAIN DF")
            #print(self.df_segment_BTC)

        self.BTCPrice_Next = 0
        self.BTCPriceAtAction = 0
        self.fullBalance = self.cashBalance
        self.prevFullBalance = self.fullBalance
        self.getInitBTCPrice()
        self.done = False
        self.cnt = 1
        self.reward = 0
        self.profit = 0
        self.previousProfit = 0
        self.firstPurchase = True

    def getInitBTCPrice(self):
        endIndex = self.endIndex
        endDate = self.df_BTC.index[endIndex]
        nextRow = self.df_BTC.loc[[endDate]]
        self.BTCPrice_Next = nextRow["Close"][0]

    def randomChart(self):
        if self.timeStepSize == "H":
            startIndex = randint((self.timeFrame + self.gameLength), (self.dataSize - 1))
            endIndex = startIndex - self.timeFrame + 1
        startDate = self.df_BTC.index[startIndex]
        endDate = self.df_BTC.index[endIndex]

        if self.timeStepSize == "H":
            startDateStr = startDate.strftime("%Y-%m-%d %H:%M:%S")
            endDateStr = endDate.strftime("%Y-%m-%d %H:%M:%S")

        return startDateStr, endDateStr, startIndex, endIndex

    def nextStep(self, action):

        self.cnt = self.cnt + 1
        self.BTCPercentChange = 0

        self.previousProfit = self.profit
        self.BTCPriceAtAction = self.BTCPrice_Next

        self.endIndex = self.endIndex - 1
        self.endDate = self.df_BTC.index[self.endIndex]
        
        self.nextRow_BTC = self.df_BTC.loc[[self.endDate]]
        self.df_segment_BTC = pd.concat([self.nextRow_BTC, self.df_segment_BTC])
        self.df_segment_BTC = self.df_segment_BTC.drop(self.df_segment_BTC.index[len(self.df_segment_BTC) - 1])

        self.nextRow_ETH = self.df_ETH.loc[[self.endDate]]
        self.df_segment_ETH = pd.concat([self.nextRow_ETH, self.df_segment_ETH])
        self.df_segment_ETH = self.df_segment_ETH.drop(self.df_segment_ETH.index[len(self.df_segment_ETH) - 1])

        self.BTCPrice_Next = self.nextRow_BTC["Close"][0]

        # --------------------------- APPLY ACTION THAT WAS TAKEN BASED ON PREVIOUS STATE ----------------------------
        if action == 1: #1
            self.actionTaken = 1
            tradeCost = self.poundsToTrade
            tradingFee = tradeCost * 0.003

            tradeCostInBTC = (tradeCost - tradingFee) / self.BTCPriceAtAction

            if self.firstPurchase == False:
                if tradeCost <= self.cashBalance:
                    self.cashBalance = self.cashBalance - tradeCost
                    self.BTC_Balance = round((self.BTC_Balance + tradeCostInBTC), 5)
                else:
                    moneyEnoughForThisBTC = self.cashBalance / self.BTCPrice_Next
                    self.cashBalance = self.cashBalance - moneyEnoughForThisBTC
                    self.BTC_Balance = round((self.BTC_Balance + moneyEnoughForThisBTC), 5)

            if self.firstPurchase == True:
                self.firstPurchase = False

                self.BTC_Balance = tradeCostInBTC
                self.cashBalance = self.cashBalance - (tradeCost)

        if action == 2: #2
            self.actionTaken = 2
            leftOverBTC = self.BTC_Balance
            self.BTC_Balance = self.BTC_Balance - leftOverBTC
            tradingFee = leftOverBTC * 0.003

            self.cashBalance = self.cashBalance + ((leftOverBTC - tradingFee) * self.BTCPrice_Next)

        if action == 0 or action == 3: #0 3
            self.actionTaken = 3

        self.cashBalance = round((self.cashBalance), 4)
        self.BTC_Balance = round((self.BTC_Balance), 8)
        self.fullBalance = round((self.cashBalance + (self.BTC_Balance * self.BTCPrice_Next)), 4)
        self.profit = self.fullBalance - self.initialBalance

        # ----------------------------------------  REWARDING SYSTEM  ---------------------------------------------
        BTCPercentGainLoss = (self.BTCPrice_Next / self.BTCPriceAtAction)
        self.BTCPercentChange = -1 * (np.round((100 - (BTCPercentGainLoss * 100)), 4))

        if self.realMoneySpent == 0:
            self.percentProfitReward = 0
        else:
            self.percentProfitReward = (self.profit / self.realMoneySpent) * 100

        # WHEN BTC WENT UP
        self.guessCnt += 1
        if self.BTCPrice_Next > self.BTCPriceAtAction:
            if self.actionTaken == 1:
                self.guessUpCnt += 1
                self.guessedRightCnt += 1
                self.guessOutcome = 1

            if self.actionTaken == 2:
                self.guessDownCnt += 1
                self.guessedWrongCnt += 1
                self.guessOutcome = -1

            if self.actionTaken == 3 or self.actionTaken == 0:
                self.guessSkipCnt += 1
                self.guessOutcome = 0

        # WHEN BTC DROPPED
        if self.BTCPrice_Next < self.BTCPriceAtAction:
            if self.actionTaken == 1:
                self.guessUpCnt += 1
                self.guessedWrongCnt += 1
                self.guessOutcome = -1

            if self.actionTaken == 2:
                self.guessDownCnt += 1
                self.guessedRightCnt += 1
                self.guessOutcome = 1

            if self.actionTaken == 3 or self.actionTaken == 0:
                self.guessSkipCnt += 1
                self.guessOutcome = 0

        # -------------------------------------- GAME ENDS IF THESE ARE MET -------------------------------------------
        if self.cnt == self.gameLength:
            self.done = True

        # -------------------------------- SELL ALL BTC AND LOCK IN PROFIT OR LOSS ------------------------------------
        if self.actionTaken == 1:
            self.reward = 0
            self.realMoneySpent += self.poundsToTrade
            self.priceAtTimeOfPurchase = self.BTCPrice_Next

        if self.actionTaken == 2:
            self.reward = self.profit
            self.profitSum += self.profit
            self.rewardSum += self.profit
            self.initialBalance = self.fullBalance
            #self.profit = 0
            #self.realMoneySpent = 0

        if self.actionTaken == 3:
            self.reward = 0

        # ---------------------- SUM UP HOW MUCH BTC MOVED PERCENTAGE WISE WHEN A CHOICE WAS MADE ----------------------
        if self.BTCPercentChange > 0:
            self.sumRightPerc += self.BTCPercentChange
        elif self.BTCPercentChange < 0:
            self.sumWrongPerc += self.BTCPercentChange

        # #print("trading fee", tradingFee)
        # print("ACTION TAKEN", self.actionTaken)
        # print("CURRENT PRICE", self.BTCPriceAtAction)
        # print("Spent:", self.realMoneySpent, "|Worth:", np.round((100 /  self.BTCPriceAtAction), 4), "BTC")
        # print("Spent: 99.7|Worth:", np.round((99.7 / self.BTCPriceAtAction), 4), "BTC")
        # print("BTC  Bal:", self.BTC_Balance, " worth= $", (self.BTC_Balance * self.BTCPriceAtAction) )
        # print("Fiat Bal:", self.cashBalance)
        # this = round((self.cashBalance + (self.BTC_Balance * self.BTCPriceAtAction)), 4)
        # print("Sum  Bal:", this)
        # #print("Ini  Bal:", self.initialBalance)
        # print("ONE HOUR LATER")
        # print("BTC Price|", self.BTCPriceAtAction, "-->", self.BTCPrice_Next, "|", self.BTCPercentChange, "%")
        # print("Profit:", self.profit)
        # print("Sum  Bal:", self.fullBalance)

        # ---------------------- ZERO OUT ----------------------
        self.BTCPriceAtAction = self.BTCPrice_Next
        if self.actionTaken == 2:
            self.profit = 0
            self.realMoneySpent = 00


        # ---------------------- ACTION LOG ----------------------
        if self.actionLogOn == True:
            if self.actionTaken == 1:
                bought = self.BTCPriceAtAction
                sold = 0

            if self.actionTaken == 2:
                bought = 0
                sold = self.BTCPriceAtAction

            if self.actionTaken == 3:
                bought = 0
                sold = 0

            self.aLogCnt += 1
            self.df_actionLog.loc[self.aLogCnt] = self.BTCPriceAtAction, bought, sold
            self.df_actionLog.to_csv(self.actionLogPathName, index=True)

        print("reward", self.reward)
        print("profit", self.reward)

        if self.done == True:
            #print("END")
            self.gamesPlayedCnt += 1
            self.profitSum += self.profit
            self.sumPercent = self.sumRightPerc + self.sumWrongPerc
            self.sumRightPerc = 0
            self.sumWrongPerc = 0
            self.profit = 0


        print("\n")

        # ------------------------------------------  WRITE EVALUATION LOG  ------------------------------------------
        if self.done == True:
            if self.evalType == "evalReal":
                df = pd.DataFrame(columns=["sumPercent", "rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "eps", "frame"])
                df.loc[self.eLogCnt] = self.sumPercent, self.rewardSum, self.profitSum, self.guessedRightCnt, self.guessedWrongCnt, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt, self.epsilon, self.frameNumber
                df.to_csv(self.evalLogPathName, mode='a', header=False)
                self.eLogCnt += 1

            if self.evalType == "evalOverfit":
                df = pd.DataFrame(columns=["sumPercent", "rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "eps", "frame"])
                df.loc[self.oLogCnt] = self.sumPercent, self.rewardSum, self.profitSum, self.guessedRightCnt, self.guessedWrongCnt, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt, self.epsilon, self.frameNumber
                df.to_csv(self.evalOverfitLogPathName, mode='a', header=False)
                self.oLogCnt += 1

            if self.evalType == "evalTrain":
                df = pd.DataFrame(columns=["sumPercent", "rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt", "eps", "frame"])
                df.loc[self.tLogCnt] = self.sumPercent, self.rewardSum, self.profitSum, self.guessedRightCnt, self.guessedWrongCnt, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt, self.epsilon, self.frameNumber
                df.to_csv(self.trainLogPathName, mode='a', header=False)
                self.tLogCnt += 1


        # ---------------------------------------------  SAVE IMAGE   ------------------------------------------------
        if 1 == 2:
            plt.imshow(image, cmap='hot')

            if self.actionTaken == 1 and self.guessOutcome == 1:
                fileName = "/home/andras/PycharmProjects/TradingGame/examination/right/guessed_up/" + str(self.guessedRightCnt) + "_" + str(np.round(self.BTCPercentChange, 2)) + "%" + ".png"
                plt.savefig(fileName)
            elif self.actionTaken == 1 and self.guessOutcome == -1:
                fileName = "/home/andras/PycharmProjects/TradingGame/examination/wrong/guessed_up/" + str(self.guessedWrongCnt) + "_" + str(np.round(self.BTCPercentChange, 2)) + "%" + ".png"
                plt.savefig(fileName)

            if self.actionTaken == 2 and self.guessOutcome == 1:
                fileName = "/home/andras/PycharmProjects/TradingGame/examination/right/guessed_down/" + str(self.guessedRightCnt) + "_" + str(np.round(self.BTCPercentChange, 2)) + "%" + ".png"
                plt.savefig(fileName)
            elif self.actionTaken == 2 and self.guessOutcome == -1:
                fileName = "/home/andras/PycharmProjects/TradingGame/examination/wrong/guessed_down/" + str(self.guessedWrongCnt) + "_" + str(np.round(self.BTCPercentChange, 2)) + "%" + ".png"
                plt.savefig(fileName)

        image = self.getChartImage(self.timeFrame)

        return image, self.reward, self.done


    def getBTCPercentChange(self):
        return self.BTCPercentChange

    def getActionTaken(self):
        return self.actionTaken

    def getProfit(self):
        return (self.fullBalance - self.initialBalance)

    def getChartData(self):
        image = self.getChartImage(self.timeFrame)
        return image

    def getCash(self):
        return self.cashBalance

    def getBTC(self):
        return self.BTC_Balance


    # CHART IMAGE GENERATION -----------------------------------------------------------------------------------------

    def getChartImage(self, timeFrame):
        
        def scale_list(x, to_min, to_max):
            def scale_number(unscaled, to_min, to_max, from_min, from_max):
                return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min

            if len(set(x)) == 1:
                print("SET(X) == 1")
                return [np.floor((to_max + to_min) / 2)] * len(x)
            else:
                return [scale_number(i, to_min, to_max, min(x), max(x)) for i in x]

        timeFrame = timeFrame
        PRICE_RANGE = timeFrame
        half_scale_size = int(PRICE_RANGE / 2)
        #half_scale_size = int(PRICE_RANGE)
        
        closes_BTC = self.df_segment_BTC["Close"]
        roundedCloses = ['%.2f' % elem for elem in closes_BTC]
        closes_BTC = closes_BTC[::-1]
        close_data_together_BTC = list(np.round(scale_list(closes_BTC[timeFrame - timeFrame: timeFrame], 0, half_scale_size - 1), 0))
        graph_close_BTC = close_data_together_BTC[0:PRICE_RANGE]

        #print(df_segment_ETH)

        closes_ETH = self.df_segment_ETH["Close"]
        roundedCloses = ['%.2f' % elem for elem in closes_ETH]
        closes_ETH = closes_ETH[::-1]
        close_data_together_ETH = list(np.round(scale_list(closes_ETH[timeFrame - timeFrame: timeFrame], 0, half_scale_size - 1), 0))
        graph_close_ETH = close_data_together_ETH[0:PRICE_RANGE]

        def graphRender(data):
            blank_matrix_close = np.zeros(shape=(half_scale_size, timeFrame))
            #print(blank_matrix_close)
            x_ind = 0
            previous_pixel = 0

            for next_pixel in data:
                blank_matrix_close[int(next_pixel), x_ind] = 215
                plus = True

                if x_ind == 0:
                    previous_pixel = next_pixel

                difference = int((previous_pixel - next_pixel))
                absDifference = abs(difference)
                previous_pixel = next_pixel

                for diff in range(absDifference):

                    if difference >= 0:
                        next_pixel = (next_pixel + 1).astype(np.uint8)
                        blank_matrix_close[next_pixel, x_ind] = 100

                    if difference < 0:
                        next_pixel = (next_pixel - 1).astype(np.uint8)
                        blank_matrix_close[next_pixel, x_ind] = 160
                x_ind += 1
            blank_matrix_close = blank_matrix_close[::-1]
            return  blank_matrix_close

        BTC = graphRender(graph_close_BTC)
        ETH = graphRender(graph_close_ETH)
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
            yAxis = reMap(self.percentProfitReward, -3,3, 84,0, 84, 0)
            radius = reMap(self.realMoneySpent, 0,2000,1,10, 10, 1)

            rr, cc = draw.circle(yAxis, 42, radius=radius, shape=matrix.shape)
            matrix[rr, cc] = 100

            c = 255 - matrix
            np.putmask(chart, c < chart, c)
            chart += matrix

            return chart

        stackedCharts = profitMeter(stackedCharts)

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
        plt.imshow(df_segment_BTC, cmap='hot')

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
        plt.imshow(df_segment_BTC, cmap='hot')
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
        #printBalances()
        plt.imshow(chart, cmap='hot')

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


def printEndReason():
    profit = test.getProfit()

    if profit < 0:
        print("----------------------------------------------------------------------------")
        print("-----------------------------          ------------------------------------")
        print("----------------------------- BANKRUPT ------------------------------------")
        print("-----------------------------          ------------------------------------")
        print("----------------------------------------------------------------------------")
    else:
        print("----------------------------------------------------------------------------")
        print("-----------------------------          ------------------------------------")
        print("--------------------------- YOU MADE MONEY ---------------------------------")
        print("-----------------------------          ------------------------------------")
        print("----------------------------------------------------------------------------")


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
    #print(df_segment_BTC)
    #printBalances()
    plt.imshow(df_segment_BTC, cmap='hot')

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



