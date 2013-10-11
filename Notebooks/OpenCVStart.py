
# In[32]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


# In[14]:

image = cv2.imread("/tmp/im.png")
im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.gray()
plt.imshow(im)


# Out[14]:

#     <matplotlib.image.AxesImage at 0x33d8d50>

# image file:

# In[ ]:




# In[12]:

print im
np.max(im)


# Out[12]:

#     [[0 0 0 ..., 0 0 0]
#      [0 0 0 ..., 0 0 0]
#      [0 0 0 ..., 0 0 0]
#      ..., 
#      [0 0 0 ..., 0 0 0]
#      [0 0 0 ..., 0 0 0]
#      [0 0 0 ..., 0 0 0]]
# 

#     255

# In[30]:

mask = np.zeros(im.shape)
contours, hier = cv2.findContours(np.array(im), 
                                  cv2.RETR_LIST, 
                                  cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, width, height = cv2.boundingRect(cnt)
    if height >= 10 and width >= 100:
        cv2.drawContours(mask, [cnt], 0, 255, 1)
plt.imshow(mask)


# Out[30]:

#     <matplotlib.image.AxesImage at 0xc584050>

# image file:

# In[34]:

im2 = ndimage.uniform_filter(im, (50, 50))
plt.imshow(im2)


# Out[34]:

#     <matplotlib.image.AxesImage at 0xcbee1d0>

# image file:
