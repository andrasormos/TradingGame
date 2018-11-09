import matplotlib.pyplot as plt
import numpy as np

Fitting_Log = np.polyfit(np.array(np.log(length)), np.array(np.log(time)), 1)
Slope_Log_Fitted = Fitting_Log[0]

Plot_Log = plt.plot(length, time, '--')
plt.xscale('log')
plt.yscale('log')
plt.show()