import numpy as np
import cv2
from matplotlib import pyplot as plt


def std_filter(I, ksize):
    F = np.ones((ksize, ksize), dtype=np.float_) / (ksize * ksize)

    MI = cv2.filter2D(I, -1, F)  # apply mean filter on I

    I2 = I * I  # I squared
    MI2 = cv2.filter2D(I2, -1, F)  # apply mean filter on I2

    return np.sqrt(MI2 - MI * MI)


def zero_crossing(I):
    """Finds locations at which zero-crossing occurs, used for
    Laplacian edge detector"""

    Ishrx = I.copy()
    Ishrx[:, 1:] = Ishrx[:, :-1]

    Ishdy = I.copy()
    Ishdy[1:, :] = Ishdy[:-1, :]

    ZC = (I == 0) | (I * Ishrx < 0) | (I * Ishdy < 0);  # zero crossing locations

    SI = std_filter(I, 3) / I.max()

    Mask = ZC & (SI > .1)

    E = Mask.astype(np.uint8) * 255  # the edges

    return E

# Load the image in grayscale
image = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
El = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=5)
log_image = zero_crossing(El)

# Get image dimensions
rows, cols = image.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

# Create figure with three subplots
fig = plt.figure(figsize=(18, 6))

# Plot Original Image Surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, image, cmap='gray', edgecolor='none')
ax1.set_title("Original Image")
ax1.set_xlabel("X axis (Width)")
ax1.set_ylabel("Y axis (Height)")
ax1.set_zlabel("Pixel Intensity")

# Plot Gaussian Blurred Image Surface
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, blurred_image, cmap='gray', edgecolor='none')
ax2.set_title("Gaussian Blurred Image")
ax2.set_xlabel("X axis (Width)")
ax2.set_ylabel("Y axis (Height)")
ax2.set_zlabel("Pixel Intensity")

# Plot Second Derivative (Laplacian of Gaussian)
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, log_image, cmap='gray', edgecolor='none')
ax3.set_title("Second Derivative (LoG)")
ax3.set_xlabel("X axis (Width)")
ax3.set_ylabel("Y axis (Height)")
ax3.set_zlabel("Pixel Intensity")

# Show the plots
plt.show()
