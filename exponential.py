import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float_kind':'{:25f}'.format})

def func(x, adj1,adj2):
    return (((x+adj1) ** pw) * adj2).astype(int)

x = [0, 16] # two given datapoints to which the exponential function with power pw should fit
y = [2,8000]

pw = 4
A = np.exp(np.log(y[0]/y[1])/pw)
a = (x[0] - x[1]*A)/(A-1)
b = y[0]/(x[0]+a)**pw

xf = np.linspace(0,16,16)
yCalcualted = func(xf, a, b)

print(yCalcualted)

plt.figure()
plt.plot(x, y, 'ko', label="Original Data")
plt.plot(xf, yCalcualted, 'r-', label="Fitted Curve")
plt.show()