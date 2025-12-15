import cv2
import numpy as np
import matplotlib.pyplot as plt

def invert_grayscale_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return
    
    inverted_image = 255 - image
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(inverted_image, cmap='gray')
    plt.title('Inverted Image')
    plt.axis('off')
    
    plt.show()

image_path = 'inversion.png'  
invert_grayscale_image(image_path)
