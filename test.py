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




y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
n = [58, 651, 393, 203, 123]

# fig, ax1 = plt.subplots()
# ax1.scatter(z, y)

xi = [i for i in range(0, len(y))]



fig = plt.figure(figsize=(12, 10))


ax1 = fig.add_subplot(111)
ax1.plot(xi, y, ".", color='b', markersize=10)


for i, txt in enumerate(n):
	print(type(txt))
	ax1.annotate(txt, (xi[i], y[i]), size=15, fontweight='bold', color='orange')


plt.show()