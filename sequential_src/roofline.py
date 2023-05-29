import matplotlib.pyplot as plt
import numpy as np

# roofline model parameters
pi = 537.6
beta = 76.8
# x values
x_values = [10**i for i in np.linspace(-2, 2, 100)]
# y values
y_values = [min(pi, beta*x) for x in x_values]

# setup the plot
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.01, 100])
ax.set_ylim([1, 3500])
ax.set_xlabel('Operational Intensity (Flop/Byte)')
ax.set_ylabel('Performance (GFlop/s)')
plt.title('Roofline ceiling for the Intel Xeon E5-2683v4 CPU\n(16 cores)')

# plot the roofline ceilings
ax.plot(x_values, y_values, 'r')
I_b = pi/beta
ax.scatter(I_b, pi, c='r')
ax.annotate('(7.0, 537.6)', xy=(I_b*0.18, pi*0.88), textcoords='data')
ax.scatter(I_b, 1.02, c='r', label='Operational intensity at the ridge point')
ax.axvline(x=I_b, ymax=0.77, color='gray', linestyle='--')
ax.text(I_b, 1, ' $I_b$', ha='left', va='bottom', fontsize=15)

# label the operational intensity of calculating U
I_U = 0.07
ax.scatter(I_U, 1.02, c='b', label='Operational intensity of calculating U')
ax.axvline(x=I_U, ymax=0.2, color='gray', linestyle='--')
# ax.annotate('0.07', xy=($I_U$, 1), textcoords='data')
ax.text(I_U, 1, '  $I_U$', ha='left', va='bottom', fontsize=15)

# label the operational intensity of the power method
I_power = 1/12
ax.scatter(I_power, 1.02, c='g', label='Operational intensity of the power method')
ax.axvline(x=I_power, ymax=0.22, color='gray', linestyle='--')
# ax.annotate('0.67', xy=($I_{power}$, 1), textcoords='data')
ax.text(I_power, 1, '$I_{power}$   ', ha='right', va='bottom', fontsize=15)

# add xticks
# ax.set_xticks([1e-2, 1e-1*1.1, 1, 10, I_b, I_U, I_power])
# ax.set_xticklabels(['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$7$', '$0.076$', '$0.83$'])

# add legends
ax.legend(loc='upper left')

# display and save the plot
plt.show()
plt.savefig('roofline.png')