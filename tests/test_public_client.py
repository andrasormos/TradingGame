import gdax
import numpy as np
import dateutil.parser
import datetime
from datetime import datetime
import matplotlib.pyplot as plt

client = gdax.PublicClient()

# output = client.get_products()
# print("get_products()")
# print(output, "\n")
#
# output = client.get_product_order_book(gdax.BTC_GBP, 1)
# print("get_product_order_book()")
# print(output, "\n")
#
output = client.get_product_ticker(gdax.ETH_USD)
print("get_product_ticker()")
print(output, "\n")
#
# output = client.get_trades(gdax.BTC_USD)
# print("get_trades()")
# print(output, "\n")

#output = client.get_historic_rates(gdax.BTC_GBP, "2018-10-01", "2018-10-07", granularity=3600)
# output = client.get_historic_rates(gdax.BTC_GBP, "2018-10-07T16:00:00", "2018-10-07T20:00:00", granularity=3600)
# print("get_historic_rates()")
# print(output, "\n")
#
# close = np.asarray(output)
# temp = close[:, [4]]
#temp = close[[0], [0]]
#print(temp)

# UNIX TO DATE -------------------------------------------------------------------------
# print("UNIX TO DATE")
# print(datetime.datetime.fromtimestamp(int("1538938800")).strftime('%Y-%m-%d %H:%M:%S'))

# # DATE TO UNIX -------------------------------------------------------------------------
# date = datetime.strptime('Sat, 06 Oct 2018 15:00:00', '%a, %d %b %Y %H:%M:%S')
# temp = date.isoformat()
# print("my converted",temp, "\n")



# output = client.get_currencies()
# print("get_currencies()")
# print(output, "\n")



output = client.time()
print("time()")
print(output, "\n")




# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(temp, "-", color='g', linewidth=1)
#
# #plt.axhline(50, color='black', linewidth=0.5)
#
# plt.show()