#from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np

from scipy import interpolate

from scipy import ndimage

y=[ 191.78 ,   191.59,    191.59,    191.41,    191.47,    191.33,    191.25  \
  ,191.33 ,   191.48 ,   191.48,    191.51,    191.43,    191.42,    191.54    \
  ,191.5975,  191.555,   191.52 ,   191.25 ,   191.15  ,  191.01  ]
x = np.linspace(1, 20, len(y))

# convert both to arrays
x_sm = np.array(x)
y_sm = np.array(y)
# print(len(x_sm))
# print(len(y_sm))

sigma = 5
x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)
print(x_g1d)

fig, ax = plt.subplots(figsize=(10, 10))
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

plt.plot(x_sm, y_sm, 'green', linewidth=1)



plt.plot(x_g1d, y_g1d, 'magenta', linewidth=1)

plt.show()