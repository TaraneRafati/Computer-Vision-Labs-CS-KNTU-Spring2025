import numpy as np
import cv2

# Load the noisy image in grayscale and normalize to [0,1]
I = cv2.imread('noisy_gaussian.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

m = 13  # Filter size (try different values, e.g., 3, 13, 21)

# === TODO: Create a 1D Gaussian kernel using cv2.getGaussianKernel ===
# Use sigma=0 to let OpenCV choose sigma automatically.
g1d = cv2.getGaussianKernel(m, sigma=0)

# === TODO: Create a 2D Gaussian kernel by taking the outer product of g1d with its transpose ===
Gkernel = g1d * g1d.T

print("1D Gaussian kernel (m=%d):" % m, g1d.flatten())
print("Sum of 1D Gaussian kernel:", g1d.sum())
print("2D Gaussian kernel sum:", Gkernel.sum())

# === TODO: Apply the Gaussian filter using cv2.filter2D (or cv2.GaussianBlur) ===
J_gauss = cv2.GaussianBlur(I, (m, m), sigmaX=0, sigmaY=0)

# === TODO: Convert the result to uint8 and save or display it ===
J_gauss = (J_gauss * 255).astype(np.uint8)
cv2.imwrite('blurred_GaussianBlur.png', J_gauss)
