import cv2
import numpy as np

I = cv2.imread('lenna.png') # load color image
I = I.astype(np.float32)/255.0

# Add noise for testing
noise = np.random.randn(*I.shape) * 0.05
noisy_color = np.clip(I + noise, 0.0, 1.0)

noisy_color_uint8 = (noisy_color * 255).astype(np.uint8)

# Bilateral filter parameters
d = 13 # diameter of pixel neighborhood (if set to >0). If =0, it uses sigmaSpace to determine.
sigma_color = 0.5
sigma_space = 50

denoised_bilateral = cv2.bilateralFilter(noisy_color_uint8 , d, sigma_color, sigma_space)
cv2.imwrite('noisy_color.png', noisy_color_uint8)
cv2.imwrite('denoised_bilateral.png', denoised_bilateral)


difference = cv2.absdiff(noisy_color_uint8, denoised_bilateral)
if np.count_nonzero(difference) == 0:
    print("Images are exactly the same.")
else:
    print("Images are different.")
print()



# import cv2
# import numpy as np

# # Load color image
# I = cv2.imread('lenna.png')  # Load color image
# I = I.astype(np.float32) / 255.0  

# # Add Gaussian noise for testing
# noise = np.random.randn(*I.shape) * 0.05
# noisy_color = np.clip(I + noise, 0.0, 1.0)

# noisy_color_uint8 = (noisy_color * 255).astype(np.uint8)  

# # Bilateral filter parameters
# d = 9  # diameter of pixel neighborhood (if set to >0). If =0, it uses sigmaSpace to determine.
# sigma_color = 0.1 * 255  
# sigma_space = 15  

# denoised_bilateral = cv2.bilateralFilter(noisy_color_uint8, d, sigma_color, sigma_space)

# cv2.imwrite('denoised_bilateral.png', denoised_bilateral)