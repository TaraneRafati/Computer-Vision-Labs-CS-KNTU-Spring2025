import cv2
import numpy as np
from matplotlib import pyplot as plt

fname = 'crayfish.jpg'
#fname = 'office.jpg'

I = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

# a = 100
# b = 175
# a = I.min()
# b = I.max()
a = np.percentile(I, 5) 
b = np.percentile(I, 95) 

J = (I - a) * 255.0 / (b - a)
J[J < 0] = 0
J[J > 255] = 255
J = J.astype(np.uint8)

K = cv2.equalizeHist(I)

f, axes = plt.subplots(2, 3)

axes[0, 0].imshow(I, 'gray', vmin=0, vmax=255)
axes[0, 0].axis('off')
axes[1, 0].hist(I.ravel(), 256, [0, 256])
axes[1, 0].set_ylim(0, 4000)

axes[0, 1].imshow(J, 'gray', vmin=0, vmax=255)
axes[0, 1].axis('off')
axes[1, 1].hist(J.ravel(), 256, [0, 256])
axes[1, 1].set_ylim(0, 4000)

axes[0, 2].imshow(K, 'gray', vmin=0, vmax=255)
axes[0, 2].axis('off')
axes[1, 2].hist(K.ravel(), 256, [0, 256])
axes[1, 2].set_ylim(0, 4000)

plt.show()