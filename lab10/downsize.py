import numpy as np
import cv2
from matplotlib import pyplot as plt

I = cv2.imread('toosi.jpg')      # Load the image
s = 4                           # downsize factor

# Downsize by sampling every s-th pixel (naive downsampling)
J = I[::s, ::s, :]              # pick every s-th pixel in both dimensions

# Initialize Jb and Jg as copies of the naive downsample (they will be replaced after blur)
Jb = J
Jg = J

# Blur with a box filter, then downsample (anti-aliasing)
ksize = s + 1
Ib = cv2.boxFilter(I, -1, (ksize, ksize))
Jb = Ib[::s, ::s, :]

# Blur with a Gaussian filter, then downsample (anti-aliasing)
sigma = (s+1)/np.sqrt(12)  # roughly equivalent sigma for Gaussian kernel of size s+1
Ig = cv2.GaussianBlur(I, (0,0), sigma)
Jg = Ig[::s, ::s, :]

J_nearest = cv2.resize(I, None, fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
J_area = cv2.resize(I, None, fx=1/s, fy=1/s, interpolation=cv2.INTER_AREA)

# Prepare a figure to display results
f, ax = plt.subplots(2, 3, gridspec_kw={'height_ratios': [s, 1]})
for a in ax.ravel():
    a.axis('off')  # turn off axes for all subplots

# Show the original image (top row, center)
ax[0,1].set_title('Original')
ax[0,1].imshow(I[:, :, ::-1])  # convert BGR to RGB for displaying with plt

# Show the downsampled images (bottom row)
ax[1,0].set_title('Downsized (naive)')
ax[1,0].imshow(J[:, :, ::-1], interpolation='none')  # no interpolation to emphasize raw pixels

ax[1,1].set_title('Box Blur + Downsized')
ax[1,1].imshow(Jb[:, :, ::-1], interpolation='none')

ax[1,2].set_title('Gaussian Blur + Downsized')
ax[1,2].imshow(Jg[:, :, ::-1], interpolation='none')

f2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
ax2[0].set_title('cv2.resize INTER_NEAREST')
ax2[0].imshow(J_nearest[:, :, ::-1], interpolation='none')
ax2[0].axis('off')

ax2[1].set_title('cv2.resize INTER_AREA')
ax2[1].imshow(J_area[:, :, ::-1], interpolation='none')
ax2[1].axis('off')

plt.tight_layout()
plt.show()
