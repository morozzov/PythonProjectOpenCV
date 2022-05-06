import cv2
import numpy as np

imgOriginal = cv2.imread("wood.jpg")
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

minThreshold = 100  # 120
maxThreshold = 255

ret, imgThreshold = cv2.threshold(imgGray, minThreshold, maxThreshold, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imgContours2 = np.zeros(imgOriginal.shape)
cv2.drawContours(imgContours2, contours, -1, (0, 0, 255), 2)

countContours = 0
for c in contours:
    area = cv2.contourArea(c)
    if 500 <= area <= 5000:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(imgContours2, (x, y), (x + w, y + h), (255, 0, 0), 2)
        countContours += 1

print("countContours = ", countContours)

imgContours = np.zeros(imgOriginal.shape)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

cv2.imshow("imgOriginal", imgOriginal)
cv2.imshow("imgGray", imgGray)
cv2.imshow("imgThreshold", imgThreshold)
cv2.imshow("imgContours", imgContours)
cv2.imshow("imgContours2", imgContours2)

cv2.waitKey(0)