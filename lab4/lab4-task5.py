import cv2
import numpy as np

# Assume sp_noisy_img is a uint8 grayscale image with salt-and-pepper noise
sp_noisy_img = cv2.imread('noisy_saltpepper.png', cv2.IMREAD_GRAYSCALE)

# Apply median filter with a 5x5 kernel
denoised_med5 = cv2.medianBlur(sp_noisy_img, 5)
# Apply median filter with a 3x3 kernel for comparison
denoised_med3 = cv2.medianBlur(sp_noisy_img, 3)
denoised_mean5 = cv2.blur(sp_noisy_img, (5, 5))

cv2.imwrite('denoised_median5.png', denoised_med5)
cv2.imwrite('denoised_median3.png', denoised_med3)
cv2.imwrite('denoised_mean5.png', denoised_mean5)
