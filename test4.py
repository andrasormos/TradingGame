from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
#y = np.sin(x)

list_y = [21.0, 17.6, 19.3, 28.9, 21.1, 20.5, 22.1, 26.4, 22.3, 24.4, 24.6, 28.0, 24.7, 24.9, 25.7, 31.6, 39.1,
			   31.3, 31.3, 32.1, 34.4, 38.0, 36.7, 39.6, 58.8, 71.8, 57.7, 62.6, 63.8, 66.3, 63.6, 81.0, 109.5, 92.7,
			   91.3, 116.0, 101.5, 96.1, 116.0, 119.1, 153.5, 162.6, 144.6, 141.5, 154.6, 174.3, 174.7, 180.6, 174.1,
			   185.2, 193.1, 196.3, 202.3, 238.6, 228.1, 228.1]
list_x = np.arange(0, len(list_y))



def f(x):
    x_points = [ 0, 1, 2, 3, 4, 5]
    y_points = [12,14,22,39,58,77]

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

print (f(1.25))