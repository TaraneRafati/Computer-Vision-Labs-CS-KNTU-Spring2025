import numpy as np
import cv2

# Load an image (using the noisy image from Task 1, or a clean image) in grayscale
I = cv2.imread('noisy_gaussian.png', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float32) / 255.0  # Normalize to [0,1]

m = 13  # Filter size (try 3, 5, 7, 11, etc.)

# === TODO: Create an m×m box filter kernel ===
# Use np.ones to create an array of ones and divide by (m*m) to normalize.
kernel = np.ones((m, m), np.float32) / (m * m)

# === TODO: Apply convolution to blur the image using cv2.filter2D ===
J = cv2.filter2D(I, -1, kernel)
J_blur = cv2.blur(I, (m, m))

# === TODO: Convert the result to uint8 and save or display it ===
# For example, use cv2.imwrite to save the result.
J_uint8 = (J * 255).astype(np.uint8)
J_blur_uint8 = (J_blur * 255).astype(np.uint8)
cv2.imwrite('blurred_filter2D.png', J_uint8)
cv2.imwrite('blurred_cv2_blur.png', J_blur_uint8)