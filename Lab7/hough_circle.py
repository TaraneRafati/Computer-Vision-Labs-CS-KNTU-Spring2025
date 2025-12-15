import cv2

I = cv2.imread('coffee.jpg')

G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY) # -> Grayscale
G = cv2.GaussianBlur(G, (3,3), 0);     # Gaussian blur

canny_high_threshold = 200
min_votes = 80 # minimum no. of votes to be considered as a circle
min_centre_distance = 20 # minimum distance between the centres of detected circles
resolution = 1 # resolution of parameters (centre, radius) relative to image resolution
circles = cv2.HoughCircles(G,cv2.HOUGH_GRADIENT,resolution,min_centre_distance,
                           param1=canny_high_threshold,
                           param2=min_votes,minRadius=0,maxRadius=100)

# for opencv 2 use cv2.cv.CV_HOUGH_GRADIENT instead of cv2.HOUGH_GRADIENT

for c in circles[0,:]:
    x = int(c[0])
    y = int(c[1])
    r = int(c[2])
   
    # draw the circle
    cv2.circle(I,(x,y), r, (0,255,0),2)

    # draw the circle center 
    cv2.circle(I,(x,y),2,(0,0,255),2)

cv2.imshow("I",I)
cv2.waitKey(0)
cv2.destroyAllWindows()
