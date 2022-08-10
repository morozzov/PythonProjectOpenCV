import cv2
import imutils as imutils
import numpy as np

imgOriginal = cv2.imread("wood.png")
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

imgBlurr = cv2.GaussianBlur(imgGray, (11, 11), 0)

minThreshold = 170  # 120
maxThreshold = 220
ret, imgThreshold = cv2.threshold(imgBlurr, minThreshold, maxThreshold, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contours = imutils.grab_contours(contours)

imgContours = np.zeros(imgOriginal.shape)
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 2)

countContours = 0
for c in contours:
    #peri = cv2.arcLength(c, True)
    #approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    area = cv2.contourArea(c)
    #if 1000 <= area <= 40000 and len(approx) > 5:
    if 1000 <= area <= 40000:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(imgContours, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (255, 0, 0), 2)
        countContours += 1
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(imgContours, "#{}".format(countContours), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        cv2.putText(imgOriginal, "#{}".format(countContours), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (50, 255, 50), 2)

print("countContours = ", countContours)

cv2.imshow("imgOriginal", imgOriginal)
cv2.imshow("imgThreshold", imgThreshold)
cv2.imshow("imgContours", imgContours)

cv2.waitKey(0)