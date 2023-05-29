import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

data = np.loadtxt('calc_svd_timing.dat')

x = data[:, 0]
y = data[:, 1]
colors = data[:, 6]

plt.scatter(x, y, c=colors, cmap='viridis', norm=LogNorm(vmin=1, vmax=8.21817e+06))

cbar = plt.colorbar()
cbar.set_label('Runtime of calculating U')

plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.title('Matrix size vs U calculation runtime')

plt.show()
