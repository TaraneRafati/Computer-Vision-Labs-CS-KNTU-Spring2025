import cv2
import numpy as np

original = cv2.imread('lenna.png').astype(np.float32) / 255.0

filter_mode = 'n'  
noise_sigma = 0.05
m = 5  
sigma_color = 25  
sigma_space = 15

def apply_filter(noisy_img_uint8, mode):
    if mode == 'n':
        return noisy_img_uint8
    elif mode == 'b':
        return cv2.blur(noisy_img_uint8, (m, m))
    elif mode == 'g':
        return cv2.GaussianBlur(noisy_img_uint8, (m, m), 0)
    elif mode == 'm':
        return cv2.medianBlur(noisy_img_uint8, m)
    elif mode == 'l':
        return cv2.bilateralFilter(noisy_img_uint8, m, sigma_color, sigma_space)
    else:
        return noisy_img_uint8

print("Controls:")
print("  'n' - No filter")
print("  'b' - Box filter")
print("  'g' - Gaussian filter")
print("  'm' - Median filter")
print("  'l' - Bilateral filter")
print("  '+' / '-' - Increase / Decrease kernel size")
print("  '.' / ',' - Increase / Decrease sigmaColor (bilateral)")
print("  'u' / 'd' - Increase / Decrease noise sigma")
print("  'q' - Quit")

while True:
    noise = np.random.randn(*original.shape) * noise_sigma
    noisy_img = np.clip(original + noise, 0, 1)
    noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)

    filtered = apply_filter(noisy_img_uint8, filter_mode)
    display = cv2.resize(filtered, (1024, 1024))
    cv2.imshow('Denoising Demo', display)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        filter_mode = 'n'
        print("Mode: No filtering")
    elif key == ord('b'):
        filter_mode = 'b'
        print("Mode: Box filter")
    elif key == ord('g'):
        filter_mode = 'g'
        print("Mode: Gaussian filter")
    elif key == ord('m'):
        filter_mode = 'm'
        print("Mode: Median filter")
    elif key == ord('l'):
        filter_mode = 'l'
        print("Mode: Bilateral filter")
    elif key == ord('+'):
        m += 2
        if m % 2 == 0: m += 1
        m = min(m, 31)
        print(f"Kernel size: {m}")
    elif key == ord('-'):
        m -= 2
        if m < 3: m = 3
        if m % 2 == 0: m -= 1
        print(f"Kernel size: {m}")
    elif key == ord('.'):
        sigma_color += 5
        print(f"sigma_color: {sigma_color}")
    elif key == ord(','):
        sigma_color = max(1, sigma_color - 5)
        print(f"sigma_color: {sigma_color}")
    elif key == ord('u'):
        noise_sigma = min(noise_sigma + 0.01, 1.0)
        print(f"Noise sigma: {noise_sigma:.3f}")
    elif key == ord('d'):
        noise_sigma = max(noise_sigma - 0.01, 0.0)
        print(f"Noise sigma: {noise_sigma:.3f}")

cv2.destroyAllWindows()
