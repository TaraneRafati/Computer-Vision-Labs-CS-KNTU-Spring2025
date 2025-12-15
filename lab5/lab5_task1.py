import numpy as np
import cv2
from matplotlib import pyplot as plt

I = cv2.imread("edge.png", cv2.IMREAD_GRAYSCALE)

# Roberts
R = np.array([[0, 1],
              [-1, 0]])
Ir = cv2.filter2D(I, cv2.CV_16S, R) 

# Prewitt
P = np.array([[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]])
Ip = cv2.filter2D(I, cv2.CV_16S, P) 

# Sobel
Dx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]) 
Ix = cv2.filter2D(I, cv2.CV_16S, Dx) 

f, axes = plt.subplots(2, 2)

axes[0,0].imshow(I, cmap = 'gray')
axes[0,0].set_title("Original Image")

axes[0,1].imshow(Ir, cmap = 'gray')
axes[0,1].set_title("Roberts")

axes[1,0].imshow(Ip, cmap = 'gray')
axes[1,0].set_title("Prewitt")

axes[1,1].imshow(Ix, cmap = 'gray')
axes[1,1].set_title("Sobel")

plt.tight_layout()
plt.show()

