import gdax
import numpy as np
import dateutil.parser
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import dateutil.parser as dp
from time import sleep
import os.path


df_BTC = pd.read_csv("./cryptoExtract/raw_ETH_USD.csv", index_col=0)


compare_date = int(df_BTC["Unix"][0])
date_index = 0
filled_df = df_BTC

for u in range(len(filled_df) - 1):

	if compare_date != int(filled_df["Unix"][date_index]):
		print(compare_date, "!=", int(filled_df["Unix"][date_index]))

	compare_date -= 900
	date_index += 1

