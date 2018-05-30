import cv2

img = cv2.imread('Screenshot_99.png')
img1 = cv2.resize(img,(400,600))
cv2.imwrite('karlo1.jpg',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
