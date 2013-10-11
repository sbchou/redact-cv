
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

image = cv2.imread("img/2605160002.png")
#grayscale the image
im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.gray()
plt.imshow(im)

print im
np.max(im)

#im.shape = size of png, as np array
mask = np.zeros(im.shape) #black mask in shape of image
contours, hier = cv2.findContours(np.array(im), 
                                  cv2.RETR_LIST, 
                                  cv2.CHAIN_APPROX_SIMPLE)

#with_dim = []

total_height, total_width = im.shape

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, width, height = cv2.boundingRect(cnt)
    #with_dim.append((area, width, height, cnt))
    #limiting this way might make sense-- that is, if we know
    #what shape censored lines might be-- aka if we're looking 
    #for full lines
    #if height >= 50 < 998:
    #    cv2.drawContours(mask, [cnt], 0, 255, 1)
    if (0.05 < height < 0.9 * total_height) and (0.05 * total_width < width < 0.9* total_width):
        cv2.drawContours(mask, [cnt], 0, 255, 1)

#with_dim = sorted(with_dim, key=lambda x:x[2], reverse=True)
#by height, since widths wouldn't vary much

#limiting this way doesn't make much sense, since who knows
#how many censorings we have?
#n = len(with_dim)
#for i in range(1, 5): #plot only top 20%
#    print with_dim[i][2]
#    cv2.drawContours(mask, [with_dim[i][3]], 0, 255, 1)

plt.imshow(mask)

#mask = np.zeros(im.shape)

#im2 = ndimage.uniform_filter(im, (50, 50))
#plt.imshow(im2)

#FUNC RETURNS THE CONTOURS
#CLEAN UP THIS SHIT AND PUSH BOTH LIBRARY CODES
#DO IT FOR GLOB
