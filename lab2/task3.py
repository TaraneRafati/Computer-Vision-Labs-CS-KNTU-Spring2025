import cv2
import numpy as np
I = cv2.imread('damavand.jpg')
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for alpha in np.linspace(0, 1, 50):
    K = cv2.addWeighted(gray, 1 - alpha, I, alpha, 0)
    cv2.imshow("Grayscale to Color", K)
    cv2.waitKey(50)
cv2.waitKey()
cv2.destroyAllWindows()
