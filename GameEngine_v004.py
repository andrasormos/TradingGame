# HUMANLY PLAYABLE PRIFUTS
from random import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox

import sys
sys.setrecursionlimit(10000) # 10000 is an example, try with different values

# Game engine v002

# .astype(np.uint8)
# np.set_printoptions(threshold=np.nan, linewidth=300)
# everything needs to be uint

class PlayGame(object):
    def __init__(self):
        self.gameIsRunning = True

        # LOAD DATA
        dateParse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
        self.df = pd.read_csv("Gdax_BTCUSD_1h_close.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)
        self.startGame()
        self.gameStep = 0

    def startGame(self):
        self.gameLength = 72  # How long the game should go on
        self.timeFrame = 84  # How many data increment should be shown as history. Could be hours, months
        self.timeStepSize = "H"  # Does nothing atm
        self.amountToSpend = 500  # How much to purchase crypto for
        self.initialBalance = 100000  # Starting Money
        self.gamePrint = False

        self.cashBalance = self.initialBalance
        self.BTC_Balance = 0  # BTC to start with
        self.actionTaken = 0

        if self.timeStepSize == "D":
            self.df = self.df.resample("D").mean()

        self.dataSize = len(self.df.index)
        self.treshMul = 70

        # GET RANDOM SEGMENT FROM DATA
        self.startDate, self.endDate, self.startIndex, self.endIndex = self.randomChart()
        self.df_segment = self.df.loc[self.startDate: self.endDate]

        self.currentBTCPrice = 0

        self.fullBalance = self.cashBalance
        self.prevFullBalance = self.fullBalance
        self.getInitBTCPrice()
        self.rekt = False
        self.done = False
        self.cnt = 1
        self.reward = 0
        self.profit = 0
        self.neededProfit = 50
        self.newProfit = 0
        self.previousProfit = 0

    def getInitBTCPrice(self):
        endIndex = self.endIndex - 1
        endDate = self.df.index[endIndex]
        nextRow = self.df.loc[[endDate]]
        self.currentBTCPrice = nextRow["Close"][0]

    def randomChart(self):
        if self.timeStepSize == "H":
            startIndex = randint((self.timeFrame + self.gameLength), (self.dataSize-1))
            endIndex = startIndex - self.timeFrame + 1

        if self.timeStepSize == "D":
            startIndex = randint((self.timeFrame + self.gameLength), (self.dataSize-1))
            endIndex = startIndex - self.timeFrame

        if self.gamePrint == True:
            print("Random Chart:", startIndex, " - ", endIndex)

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
        print("---------- AN HOUR LATER ----------")
        self.gameStep += 1
        self.cnt = self.cnt + 1
        self.reward = 0

        self.endIndex = self.endIndex - 1
        self.endDate = self.df.index[self.endIndex]
        self.nextRow = self.df.loc[[self.endDate]]

        self.df_segment = pd.concat([self.nextRow, self.df_segment])
        self.df_segment = self.df_segment.drop(self.df_segment.index[len(self.df_segment)-1])

        self.currentBTCPrice = self.nextRow["Close"][0]

        if action == 1:
            print("BOUGHT $500 WORTH BTC")
            self.actionTaken = 1
            if self.amountToSpend > self.cashBalance:
                self.cashBalance = 0
                self.BTC_Balance = round((self.BTC_Balance + (self.cashBalance / self.currentBTCPrice)), 5)
            else:
                self.cashBalance = self.cashBalance - self.amountToSpend
                self.BTC_Balance = round((self.BTC_Balance + (self.amountToSpend / self.currentBTCPrice)), 5)

        if action == 2:
            print("SOLD $500 WORTH BTC")
            moneyWorthInBTC = self.amountToSpend / self.currentBTCPrice  # 0.1

            if moneyWorthInBTC > self.BTC_Balance:
                self.cashBalance = self.cashBalance + (self.BTC_Balance * self.currentBTCPrice)
                self.BTC_Balance = 0
            else:
                self.BTC_Balance = self.BTC_Balance - moneyWorthInBTC
                self.cashBalance = self.cashBalance + self.amountToSpend

        if action == 3:
            print("SKIPPED AN HOUR")
            self.actionTaken = 0

        self.cashBalance = round((self.cashBalance), 0)
        self.BTC_Balance = round((self.BTC_Balance), 5)
        self.fullBalance = round((self.cashBalance + (self.BTC_Balance * self.currentBTCPrice)), 0)
        self.profit = self.fullBalance - self.initialBalance

        #  - REWARDING SYSTEM -
        self.neededProfit = np.round((self.treshMul * np.sqrt(self.BTC_Balance)), 0)
        self.endGameProfit = (self.treshMul * np.sqrt(self.BTC_Balance)) * 2

        self.newProfit = self.profit - self.previousProfit

        if self.newProfit > self.neededProfit:
            self.reward = 1
            self.previousProfit = self.profit

        if self.newProfit < (-1 * self.neededProfit):
            self.reward = -1
            self.previousProfit = self.profit


        if self.newProfit > self.endGameProfit:
            self.reward = 2
            self.previousProfit = self.profit
            self.done = True
            print("WON THE GAME!")

        if self.newProfit < (-1 * self.endGameProfit):
            self.reward = -1
            self.previousProfit = self.profit
            self.done = True
            print("REKT!")

        print("M:", self.previousProfit, "|")
        print("|", self.neededProfit, "|", self.newProfit, "|")

        if self.cnt == self.gameLength:
            self.done = True
            print("GAME LENGTH REACHED!")

        if self.done == True:
            self.gameStep = 0

        if self.gamePrint == True:
            if action == 1:
                print("-BOUGHT BTC-")
            if action == 2:
                print("-SOLD BTC-")
            if action == 3:
                print("-SKIP-")

            print("-",self.gameStep,"-" ,self.endDate, "- PROFIT:", self.profit, "BAL:", self.fullBalance, "BTC", self.BTC_Balance," CASH:", self.cashBalance,"BTC $:",self.currentBTCPrice)
            print("\n")
        image = self.getChartImage(self.timeFrame, self.df_segment)

        return image, self.reward, self.done

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
        #half_scale_size = int(PRICE_RANGE / 2)
        half_scale_size = int(PRICE_RANGE)
        stock_closes = df_segment["Close"]
        roundedCloses = ['%.2f' % elem for elem in stock_closes]
        stock_closes = stock_closes[::-1]
        close_data_together = list(np.round (scale_list (stock_closes[timeFrame - timeFrame : timeFrame], 0, half_scale_size - 1), 0) )
        graph_close = close_data_together[0:PRICE_RANGE]

        # TOP HALF
        blank_matrix_close = np.zeros(shape=(half_scale_size, timeFrame))

        x_ind = 0
        previous_pixel = 0

        for next_pixel in graph_close:
            blank_matrix_close[int(next_pixel), x_ind] = 1
            plus = True

            if x_ind == 0:
                previous_pixel = next_pixel

            difference = int((previous_pixel - next_pixel))
            absDifference = abs(difference)
            previous_pixel = next_pixel

            for diff in range(absDifference):


                if difference >=0:
                    next_pixel = (next_pixel + 1).astype(np.uint8)
                    blank_matrix_close[next_pixel, x_ind] = 0.2

                if difference < 0:
                    next_pixel = (next_pixel - 1).astype(np.uint8)
                    blank_matrix_close[next_pixel, x_ind] = 0.5

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
        buyButt = Button(buyCom, 'BUY', color='red', hovercolor='green')
        buyButt.on_clicked(_buy)

        sellCom = plt.axes([0.9, 0.1, 0.1, 0.075])
        sellButt = Button(sellCom, 'SELL', color='red', hovercolor='green')
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

        printBalances()
        printEndReason()
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
        test.startGame()
        HourLater(1)

    else:
        chart, r_t, terminal = test.nextStep(action)
        print("REWARDED:", r_t)
        printBalances()
        plt.imshow(chart, cmap='hot')

        buyCom = plt.axes([0.9, 0.2, 0.1, 0.075])
        buyButt = Button(buyCom, 'BUY', color='red', hovercolor='green')
        buyButt.on_clicked(_buy)

        sellCom = plt.axes([0.9, 0.1, 0.1, 0.075])
        sellButt = Button(sellCom, 'SELL', color='red', hovercolor='green')
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
    df_segment = test.getChartData()
    printBalances()
    plt.imshow(df_segment, cmap='hot')

    buyCom = plt.axes([0.9, 0.2, 0.1, 0.075])
    buyButt = Button(buyCom, 'BUY', color='red', hovercolor='green')
    buyButt.on_clicked(_buy)

    sellCom = plt.axes([0.9, 0.1, 0.1, 0.075])
    sellButt = Button(sellCom, 'SELL', color='red', hovercolor='green')
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



