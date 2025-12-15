from matplotlib import pyplot as plt
import numpy as np

I = plt.imread('masoleh_gray.jpg')
I2 = I[::-1, :]
I = np.vstack((I, I2))

plt.imshow(I,cmap='gray')
plt.show()
