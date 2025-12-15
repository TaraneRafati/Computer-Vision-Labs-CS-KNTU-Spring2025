import numpy as np
import cv2

# Load the image in grayscale and normalize to [0,1]
I = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float32) / 255.0  # Ensure pixel values are in [0,1]

noise_sigma = 0.05  # initial noise standard deviation

while True:
    # TODO: Create a noise image N using a Gaussian distribution with mean 0 and variance noise_sigma^2.
    # Hint: Use np.random.randn with the shape of I and multiply by noise_sigma.
    N = np.random.randn(*I.shape) * noise_sigma  # complete if needed

    # TODO: Add the noise to the original image and clip the result to ensure values remain in [0,1].
    noisy_image = np.clip(I+N, 0, 1)
    J = (noisy_image * 255).astype(np.uint8) 
    cv2.imshow('Snow Noise', J)

    key = cv2.waitKey(33) & 0xFF

    # TODO: Adjust noise_sigma based on key input:
    if key == ord('u'):
         # TODO: Increase noise_sigma by a small amount (e.g., 0.01)
         noise_sigma += 0.01
    elif key == ord('d'):
         noise_sigma = max(0, noise_sigma - 0.01)   
         # TODO: Decrease noise_sigma by a small amount (e.g., 0.01)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()