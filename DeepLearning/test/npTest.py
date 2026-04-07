import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3.14, 3.14, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
