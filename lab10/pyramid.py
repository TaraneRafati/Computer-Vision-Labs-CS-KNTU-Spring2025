import numpy as np
import cv2
from matplotlib import pyplot as plt

I = cv2.imread('toosi.jpg')
psize = 6  # number of levels in the pyramid

# Build the image pyramid
J = I.copy()
Pyr = [J]                  # level 0: original image
for i in range(psize - 1):
    J = cv2.pyrDown(J)     # blur and downsample by factor of 2
    Pyr.append(J)          # append the smaller image to the pyramid list


def manual_pyramid(I, psize):
    Pyr = [I.copy()]
    for i in range(psize - 1):
        blurred = cv2.GaussianBlur(Pyr[-1], (5, 5), sigmaX=1)
        downsampled = blurred[::2, ::2]  # pick every second pixel
        Pyr.append(downsampled)
    return Pyr
Pyr_manual = manual_pyramid(I, psize)


# Display the pyramid levels side by side
# Create a wide figure with subplots for each level
size_list = [2**(psize - 1 - i) for i in range(psize)]  # for nicer scaling
f, ax = plt.subplots(2, psize, gridspec_kw={'width_ratios': size_list})

for a in ax.ravel():
    a.axis('off')

# pyrDown
for l in range(psize):
    ax[0, l].set_title(f"Level {l}")
    ax[0, l].imshow(Pyr[l][:, :, ::-1], interpolation='none')

# manual 
for l in range(psize):
    ax[1, l].imshow(Pyr_manual[l][:, :, ::-1], interpolation='none')

plt.tight_layout()
plt.show()


