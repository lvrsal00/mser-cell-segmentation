import numpy
import cv2

# original slika
img = cv2.imread(
    'C:/Users/Lucija/Desktop/mser_cell_segmentation/images/Tv111.tif', 1)
cv2.imshow('original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # siva slika
cv2.imshow('gray', gray)

vis = img.copy()

ret, thresh = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('thresh', thresh)

mser = cv2.MSER_create()
#mser = cv2.MSER_create(_min_area=2, _max_area=1000)
regions = mser.detectRegions(thresh)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('mser', vis)

cv2.waitKey(0)
