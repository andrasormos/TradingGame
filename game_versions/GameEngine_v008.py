from random import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox

import sys

sys.setrecursionlimit(10000)  # 10000 is an example, try with different values

# Game engine v002

# .astype(np.uint8)
# np.set_printoptions(threshold=np.nan, linewidth=300)
# everything needs to be uint


class PlayGame(object):
    def __init__(self):
        # LOAD DATA
        dateParse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
        dateParse2 = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")

        self.training_df = pd.read_csv("/home/andras/PycharmProjects/TradingGame/crypto/Gdax_BTCUSD_1h_close_train.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)
        self.eval_df = pd.read_csv("/home/andras/PycharmProjects/TradingGame/crypto/Gdax_BTCUSD_1h_close_eval.csv", parse_dates=["Date"], date_parser=dateParse2, index_col=0)

        self.gameStep = 0

        self.rewardList = []
        self.rewardSum = 0

        self.trainLogName = "/home/andras/PycharmProjects/TradingGame/logs/trainLog_016.csv"
        self.evalLogName = "/home/andras/PycharmProjects/TradingGame/logs/evalLog_016.csv"

        self.trainLogFile = pd.DataFrame(columns=["rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt"])
        self.evalLogFile = pd.DataFrame(columns=["rewardSum", "profit", "guessedRightCnt", "guessedWrongCnt", "guessUpCnt", "guessDownCnt", "guessSkipCnt", "guessCnt"])

        self.guessedRightCnt = 0
        self.guessedWrongCnt = 0
        self.guessUpCnt = 0
        self.guessDownCnt = 0
        self.guessSkipCnt = 0
        self.guessCnt = 0
        self.eLogCnt = 0
        self.tLogCnt = 0

    def startGame(self, evaluation):
        self.evaluation = evaluation


        if self.evaluation == True:
            self.df = self.eval_df
        else:
            self.df = self.training_df


        self.gameLength = 72  # How long the game should go on
        self.timeFrame = 84  # How many data increment should be shown as history. Could be hours, months
        self.timeStepSize = "H"  # Does nothing atm
        self.amountToSpend = 500  # How much to purchase crypto for
        self.initialBalance = 100000  # Starting Money
        self.gamePrint = True

        self.cashBalance = self.initialBalance
        self.BTC_Balance = 0  # BTC to start with
        self.actionTaken = 0

        self.BTCToTrade = 0.01

        if self.timeStepSize == "D":
            self.df = self.df.resample("D").mean()

        self.dataSize = len(self.df.index)

        # GET RANDOM SEGMENT FROM DATA

        self.startDate, self.endDate, self.startIndex, self.endIndex = self.randomChart()

        if self.evaluation == True:
            self.df_segment = self.df.loc[self.endDate: self.startDate]
        else:
            self.df_segment = self.df.loc[self.startDate: self.endDate]

        # print("Random Chart:", self.startIndex, " - ", self.endIndex)
        # print("Random Chart:", self.startDate, " - ", self.endDate)

        self.currentBTCPrice = 0
        self.previousBTCPrice = 0

        self.fullBalance = self.cashBalance
        self.prevFullBalance = self.fullBalance
        self.getInitBTCPrice()

        self.done = False
        self.cnt = 1
        self.reward = 0
        self.profit = 0

        self.firstPurchase = True

    def getInitBTCPrice(self):
        endIndex = self.endIndex
        endDate = self.df.index[endIndex]
        nextRow = self.df.loc[[endDate]]
        self.currentBTCPrice = nextRow["Close"][0]

    def randomChart(self):
        if self.timeStepSize == "H":
            startIndex = randint((self.timeFrame + self.gameLength), (self.dataSize - 1))
            endIndex = startIndex - self.timeFrame + 1

        if self.timeStepSize == "D":
            startIndex = randint((self.timeFrame + self.gameLength), (self.dataSize - 1))
            endIndex = startIndex - self.timeFrame

        startDate = self.df.index[startIndex]
        endDate = self.df.index[endIndex]

        if self.timeStepSize == "H":
            startDateStr = startDate.strftime("%Y-%m-%d %H:%M:%S")
            endDateStr = endDate.strftime("%Y-%m-%d %H:%M:%S")

        if self.timeStepSize == "D":
            startDateStr = startDate.strftime("%Y-%m-%d")
            endDateStr = endDate.strftime("%Y-%m-%d")

        return startDateStr, endDateStr, startIndex, endIndex

    def nextStep(self, action):
        #print("\n")
        self.gameStep += 1
        self.cnt = self.cnt + 1
        self.reward = 0
        self.BTCPercentChange = 0
        terminal_life_lost = False

        self.previousBTCPrice = self.currentBTCPrice

        self.endIndex = self.endIndex - 1
        self.endDate = self.df.index[self.endIndex]
        self.nextRow = self.df.loc[[self.endDate]]

        self.df_segment = pd.concat([self.nextRow, self.df_segment])
        self.df_segment = self.df_segment.drop(self.df_segment.index[len(self.df_segment) - 1])

        self.currentBTCPrice = self.nextRow["Close"][0]
        #print(self.previousBTCPrice, "-->", self.currentBTCPrice)

        if action == 1:
            #print("Guess: Increase")
            self.actionTaken = 1
            if self.firstPurchase == True:
                self.firstPurchase = False
                self.BTC_Balance = 0.1
                self.cashBalance = self.cashBalance - (0.1 * self.currentBTCPrice)

            if self.firstPurchase == False:
                tradeCost = self.BTCToTrade * self.currentBTCPrice

                if tradeCost <= self.cashBalance:
                    self.cashBalance = self.cashBalance - tradeCost
                    self.BTC_Balance = round((self.BTC_Balance + self.BTCToTrade), 5)
                    #print("BOUGHT", self.BTCToTrade, "BTC for", tradeCost)
                else:
                    #print("RAN OUT OF MONEY!!!")
                    moneyEnoughForThisBTC = self.cashBalance / self.currentBTCPrice
                    self.cashBalance = self.cashBalance - moneyEnoughForThisBTC
                    self.BTC_Balance = round((self.BTC_Balance + moneyEnoughForThisBTC), 5)
                    #print("BOUGHT", moneyEnoughForThisBTC, "BTC for", self.cashBalance)

            self.BTCToTrade = self.BTCToTrade + (self.BTCToTrade * 0.3)

        if action == 2:
            #print("Guess: Decrease")
            self.actionTaken = 2
            if self.BTCToTrade <= self.BTC_Balance:
                self.BTC_Balance = self.BTC_Balance - self.BTCToTrade
                self.cashBalance = self.cashBalance + (self.BTCToTrade * self.currentBTCPrice)
                #print("SOLD", self.BTCToTrade, "BTC for", (self.BTCToTrade * self.currentBTCPrice))

            else:
                leftOverBTC = self.BTC_Balance * self.currentBTCPrice
                self.BTC_Balance = self.BTC_Balance - leftOverBTC
                self.cashBalance = self.cashBalance + (leftOverBTC * self.currentBTCPrice)
                #print("SOLD", leftOverBTC, "BTC for", (leftOverBTC * self.currentBTCPrice))

            self.BTCToTrade = self.BTCToTrade - (self.BTCToTrade * 0.3)

        if action == 0 or action == 3:
            ####print("Skipped")
            self.actionTaken = 3

        self.cashBalance = round((self.cashBalance), 0)
        self.BTC_Balance = round((self.BTC_Balance), 5)
        self.fullBalance = round((self.cashBalance + (self.BTC_Balance * self.currentBTCPrice)), 0)
        self.profit = self.fullBalance - self.initialBalance

        #  - REWARDING SYSTEM -
        BTCPercentGainLoss = (self.currentBTCPrice / self.previousBTCPrice)
        self.BTCPercentChange = -1 * (np.round((100 - (BTCPercentGainLoss * 100)), 2))

        #print(self.previousBTCPrice, "-->", self.currentBTCPrice)
        #print("changed by:", self.BTCPercentChange)
        
        # WHEN BTC WENT UP
        if self.currentBTCPrice > self.previousBTCPrice:
            if self.actionTaken == 1:
                self.reward = self.BTCPercentChange
                self.guessedRightCnt += 1
                #print("Guessed Right - reward =", self.reward)
                
            if self.actionTaken == 2:
                self.reward = self.BTCPercentChange * -1
                self.guessedWrongCnt += 1
                ##print("Guessed Wrong - reward =", self.reward)

            if self.actionTaken == 3 or self.actionTaken == 0:
                self.reward = 0

        # WHEN BTC DROPPED
        if self.currentBTCPrice < self.previousBTCPrice:
            if self.actionTaken == 1:
                self.reward = self.BTCPercentChange
                self.guessedRightCnt += 1
                #print("Guessed Wrong - reward =", self.reward)

            if self.actionTaken == 2:
                self.reward = self.BTCPercentChange * -1
                self.guessedWrongCnt += 1
                #print("Guessed Right - reward =", self.reward)

            if self.actionTaken == 3 or self.actionTaken == 0:
                self.reward = 0

        self.guessCnt += 1


        #print("\n")
        outcome = 0

        self.previousBTCPrice = self.currentBTCPrice

        if self.cnt == self.gameLength:
            self.done = True
            #print("game length reached")

        if self.done == True:
            terminal_life_lost = True
            self.gameStep = 0


        image = self.getChartImage(self.timeFrame, self.df_segment)

        if self.actionTaken == 1:
            self.guessUpCnt += 1

        if self.actionTaken == 2:
            self.guessDownCnt +=1

        if self.actionTaken == 3:
            self.guessSkipCnt +=1
        
        
        if self.evaluation == False:
            self.rewardSum = self.rewardSum + self.reward
            
            if self.done == True:
                # REWARD AND PROFIT LOG
                self.trainLogFile.loc[self.tLogCnt] = self.rewardSum, self.profit, self.guessedRightCnt, self.guessedWrongCnt, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt
                self.tLogCnt += 1
                self.trainLogFile.to_csv(self.trainLogName, index=True)

                self.rewardSum = 0
                self.guessUpCnt = 0
                self.guessDownCnt = 0
                self.guessedRightCnt = 0
                self.guessedWrongCnt = 0
                self.guessSkipCnt = 0
                self.guessCnt = 0

        else:
            self.rewardSum = self.rewardSum + self.reward
            if self.done == True:
                # REWARD AND PROFIT LOG
                self.evalLogFile.loc[self.eLogCnt] = self.rewardSum, self.profit, self.guessedRightCnt, self.guessedWrongCnt, self.guessUpCnt, self.guessDownCnt, self.guessSkipCnt, self.guessCnt
                self.eLogCnt += 1
                self.evalLogFile.to_csv(self.evalLogName, index=True)

                #print("rewardSum", self.rewardSum)
                #print("guessedRightCnt", self.guessedRightCnt)
                #print("guessedWrongCnt", self.guessedWrongCnt)
                #print("guessSkipCnt", self.guessSkipCnt)
                #print("guessCnt", self.guessCnt)

                self.rewardSum = 0
                self.guessUpCnt = 0
                self.guessDownCnt = 0
                self.guessedRightCnt = 0
                self.guessedWrongCnt = 0
                self.guessSkipCnt = 0
                self.guessCnt = 0

        return image, self.reward, self.done

    def getBTCPercentChange(self):
        return self.BTCPercentChange

    def getActionTaken(self):
        return self.actionTaken


    def getProfit(self):
        return (self.fullBalance - self.initialBalance)


    def getChartData(self):
        image = self.getChartImage(self.timeFrame, self.df_segment)
        return image


    def getCash(self):
        return self.cashBalance


    def getBTC(self):
        return self.BTC_Balance


    # ---------------------------------     CHART IMAGE GENERATION      ---------------------------------

    def getChartImage(self, timeFrame, df_segment):
        #print(df_segment)
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
        # half_scale_size = int(PRICE_RANGE / 2)
        half_scale_size = int(PRICE_RANGE)
        stock_closes = df_segment["Close"]
        roundedCloses = ['%.2f' % elem for elem in stock_closes]
        stock_closes = stock_closes[::-1]
        close_data_together = list(
            np.round(scale_list(stock_closes[timeFrame - timeFrame: timeFrame], 0, half_scale_size - 1), 0))
        graph_close = close_data_together[0:PRICE_RANGE]

        # TOP HALF
        blank_matrix_close = np.zeros(shape=(half_scale_size, timeFrame))

        x_ind = 0
        previous_pixel = 0

        for next_pixel in graph_close:
            blank_matrix_close[int(next_pixel), x_ind] = 255
            plus = True

            if x_ind == 0:
                previous_pixel = next_pixel

            difference = int((previous_pixel - next_pixel))
            absDifference = abs(difference)
            previous_pixel = next_pixel

            for diff in range(absDifference):

                if difference >= 0:
                    next_pixel = (next_pixel + 1).astype(np.uint8)
                    blank_matrix_close[next_pixel, x_ind] = 80

                if difference < 0:
                    next_pixel = (next_pixel - 1).astype(np.uint8)
                    blank_matrix_close[next_pixel, x_ind] = 180

            x_ind += 1

        blank_matrix_close = blank_matrix_close[::-1]

        '''
        # BOTTOM HALF
        blank_matrix_diff = np.zeros(shape=(half_scale_size, timeFrame))
        x_ind = 0
        for v in graph_close:
            blank_matrix_diff[int(v), x_ind] = 0
            x_ind += 1
        blank_matrix_diff = blank_matrix_diff[::-1]
    
        # TOP + BOTTOM
        blank_matrix = np.vstack([blank_matrix_close, blank_matrix_diff])
        '''
        blank_matrix = blank_matrix_close

        return blank_matrix


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
        df_segment = test.getChartData()
        plt.imshow(df_segment, cmap='hot')

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
        df_segment = test.getChartData()
        plt.imshow(df_segment, cmap='hot')
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
        test.startGame(True)
        HourLater(1)

    else:
        chart, r_t, terminal = test.nextStep(action)
        # printBalances()
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
    test.startGame(True)
    df_segment = test.getChartData()
    #print(df_segment)
    printBalances()
    plt.imshow(df_segment, cmap='hot')

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
    newGame()



