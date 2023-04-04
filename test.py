from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

im = Image.open('building.jpg')
im = np.asarray(im)

X, Y= im.shape[1], im.shape[0]

print('1. original 2. lighten 3. box filtering\n')
print('select : ', end = '')
n = int(input())

if n == 1:      # original
    pass
elif n == 2:    # lighten
    v = 100
    im = im + v
    im[im <= v] = 255
elif n == 3:    # box flitering
    k = 3
    kernel = np.ones([k, k]) / (k * k) 

    p = k // 2
    x_out, y_out = X - 2 * p, Y - 2 * p 
    output = np.zeros([y_out, x_out, im.shape[2]])

    for y in range(p, Y - p):
        for x in range(p, X - p):
            for d in range(3):  # dimension
                output[y - p, x - p, d] = np.sum(im[y - p : y + p + 1, x - p : x + p + 1, d] * kernel)

    im = output / 255.0

plt.figure()
plt.imshow(im)
plt.show()