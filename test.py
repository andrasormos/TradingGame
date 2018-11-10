import numpy as np
import matplotlib.pyplot as plt
from numpy import sin,linspace,power
from pylab import plot,show
from scipy import interpolate
from numpy import arange

# Fresh potatoes: Annual Consumer price index, 1949-2006
# obatined at https://explore.data.gov/Agriculture/U-S-Potato-Statistics/cgk7-6ccj
price_index = [21.0,17.6,19.3,28.9,21.1,20.5,22.1,26.4,22.3,24.4,24.6,28.0,24.7,24.9,25.7,31.6,39.1,31.3,31.3,32.1,34.4,38.0,36.7,39.6,58.8,71.8,57.7,62.6,63.8,66.3,63.6,81.0,109.5,92.7,91.3,116.0,101.5,96.1,116.0,119.1,153.5,162.6,144.6,141.5,154.6,174.3,174.7,180.6,174.1,185.2,193.1,196.3,202.3,238.6,228.1,231.1,247.7,273.1]
t = np.arange(0,len(price_index))



def draw_tangent(x,y,point_x):
	# interpolate the data with point_x spline
	spline = interpolate.splrep(x, y)
	line = arange(point_x - 5, point_x + 5)
	point_y = interpolate.splev(point_x, spline, der=0)     # f(point_x)
	fprime = interpolate.splev(point_x, spline, der=1)  # f'(point_x)
	tan = point_y + fprime * (line - point_x)  # tangent

	plot(point_x, point_y, 'o')
	plot(line, tan, '.') # '--r'


draw_tangent(t, price_index, 41)
plot(t, price_index, alpha=0.5)
show()


#from scipy.interpolate import splrep, splev
list_x = t
list_y = price_index


plt.figure()
bspl = interpolate.splrep(list_x, list_y, s=5000)
bspl_y = interpolate.splev(list_x, bspl)


plt.plot(list_x, list_y)
plt.plot(list_x, bspl_y)
plt.show()


price_index = price_index[:-19]
t = t[:-19]

