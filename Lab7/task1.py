import cv2

I = cv2.imread('coins.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
G = cv2.GaussianBlur(G, (5,5), 0)

canny_high_threshold = 160
min_votes = 30 
min_centre_distance = 40
min_radius = 20
max_radius = 60

circles = cv2.HoughCircles(
    G,
    cv2.HOUGH_GRADIENT,
    dp=1,                      
    minDist=min_centre_distance,
    param1=canny_high_threshold,
    param2=min_votes,
    minRadius=min_radius,
    maxRadius=max_radius
)

n = len(circles[0,:]) if circles is not None else 0
for c in circles[0,:]:
    x = int(c[0])
    y = int(c[1])
    r = int(c[2])
    cv2.circle(I,(x,y), r, (0,255,0),2)
    cv2.circle(I,(x,y),2,(0,0,255),2)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(I, f'There are {n} coins!', (30, 40), font, 1, (255, 0, 0), 2)

cv2.imshow("I",I)
cv2.waitKey(0)
