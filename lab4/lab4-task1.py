import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Load image in grayscale and convert to float [0,1]
I = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float32) / 255.0  # shape: (H, W)

# Function to add Gaussian noise
def add_gaussian_noise(img, sigma=0.05):
    # TODO: Generate a noise array using np.random.randn scaled by sigma.
    noise = np.random.randn(*img.shape) * sigma
    # TODO: Add the noise to the input image.
    noisy_img = img + noise
    # TODO: Clip the resulting values to ensure they remain in the [0,1] range.
    return np.clip(noisy_img, 0, 1), noise

# Function to add salt-and-pepper noise
def add_salt_pepper_noise(img, p=0.02):
    # TODO: Create a copy of the image to modify.
    noisy_img = img.copy()
    # TODO: Determine the number of pixels to alter based on the given p
    num_noisy = int(p * img.size)
    # TODO: Randomly choose indices for salt (set to 1.0) and pepper (set to 0.0).
    num_salt = num_noisy // 2
    num_pepper = num_noisy - num_salt
    coords_salt = [np.random.randint(0, i, num_salt) for i in noisy_img.shape]
    noisy_img[coords_salt[0], coords_salt[1]] = 1.0
    coords_pepper = [np.random.randint(0, i, num_pepper) for i in noisy_img.shape]
    noisy_img[coords_pepper[0], coords_pepper[1]] = 0.0
    return noisy_img

# Generate noisy images using your implementations
gauss_noisy, noise = add_gaussian_noise(I, sigma=0.1)
sp_noisy = add_salt_pepper_noise(I)

# Convert the noisy images back to uint8 for saving or displaying
cv2.imwrite('noisy_gaussian.png', (gauss_noisy * 255).astype(np.uint8))
cv2.imwrite('noisy_saltpepper.png', (sp_noisy * 255).astype(np.uint8))

gaussian_noise = gauss_noisy - I
plt.figure()
plt.hist(gaussian_noise.ravel(), bins=50)
plt.title(f'Histogram of Gaussian Noise')
plt.xlabel('Noise Value')
plt.ylabel('Frequency')
plt.show()

plt.figure()
plt.hist(sp_noisy.ravel(), bins=50)
plt.title(f'Histogram of Salt-and-Pepper Noise')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()