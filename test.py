import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 1000)
line = np.arange(0, 1000)
curve = np.sin(np.arange(0, 10, 0.01) * 2) * 1000

plt.plot(x, line, '-')
plt.plot(x, curve, '-')

idx = np.argwhere(np.diff(np.sign(line - curve))).flatten()
# print(line)
# print(curve)
print(idx)

plt.plot(x[idx], line[idx], 'ro')
plt.show()