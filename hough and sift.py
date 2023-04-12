#-*-coding:cp949-*-

from ast import With
from itertools import accumulate
import cv2
from matplotlib import pyplot as plt
import numpy as np

def pltShow(title, image):
    plt.figure(title)
    plt.imshow(image)
    plt.show()

'''
# hough
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
'''

# shift
im = cv2.imread('mona_source.png')
im2 = cv2.imread('mona_target.jpg')

# step-1 read the images
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imt2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# step-2 perform SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(im, None)

#k = 0
#print(keypoints[k])

imk = cv2.drawKeypoints(im, keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#pltShow('step-2', imk)

# Step-3 display features of a common keypoint
pt = (224, 225) # chosen keypoint location
kpt = [k.pt for k in keypoints]
kpts = np.array(kpt)
print(kpts - pt)

sq2_diff = (kpts - pt)**2               # sq2_diff.size : [N, 2]
sq1_diff = np.sum(sq2_diff, axis = 1)   # sq1_diff.size : [N, 1]

k = np.argmin(sq1_diff)   # pt와 가장 가까운 keypoint 탐색
               
print(keypoints[k].pt)
feats = descriptors[k]

plt.bar(np.arange(0, len(feats)), feats, width = 0.5)
plt.show()

# Step-4 display features of different keypoints

# 3번 과제 : target 이미지에서 코 주위의 keypoint로 똑같이 적용
# 4번 과제 : source, target 중 3번에서 사용한 keypoint 제외하고 같은 작업
# 각각 어떤 keypoint 선택했는지, 그에 따른 bar 스크린샷 제출
# 꼭 pdf로 올리기