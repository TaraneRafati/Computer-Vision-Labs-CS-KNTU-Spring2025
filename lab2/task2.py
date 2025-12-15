import cv2
import numpy as np
I = cv2.imread('damavand.jpg')
J = cv2.imread('eram.jpg')
for alpha in np.linspace(0, 1, 50):
    K = cv2.addWeighted(I, 1 - alpha, J, alpha, 0)
    cv2.imshow("Smooth Transition", K)
    cv2.waitKey(100)
cv2.waitKey()
cv2.destroyAllWindows()
