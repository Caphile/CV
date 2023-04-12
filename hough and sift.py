from ast import With
from itertools import accumulate
import cv2
from matplotlib import pyplot as plt

im = cv2.imread('mona_target.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imshow('input', im)
edge = cv2.Canny(im, 50, 150)

cv2.imshow('edge', edge)
cv2.waitKey(0)

#plt.imshow(im, cmap = 'gray')

# detect hough lines
import numpy as np
acc = cv2.HoughLinesWithAccumulator(edge, 1, np.pi / 180, 50)
acc = np.squeeze(acc)

print(acc.shape)

# draw the lines
im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

k = 0
r, theta = acc[k][0], acc[k][1]
a, b = np.cos(theta), np.sin(theta)
x = [0, 10000]
if b:
    (x1, y1) = (x[0], int((r - x[0] * a) / b))
    (x2, y2) = (x[1], int((r - x[1] * a) / b))
else:
    (x1, y1) = (int(r / a), x[0]) 
    (x2, y2) = (int(r / a), x[1])

cv2.line(im2, (x1, y1), (x2, y2), (0, 0, 255))  # B G R
cv2.imshow('line', im2)
cv2.waitKey(0)