import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def censor_dark(img_url, min_width_ratio, max_width_ratio,  min_height_ratio, max_height_ratio):
    """Return mask with dark censor contours.

    Parameters
    ----------
    img_url : path to image
    min_width_ratio : minimum width of censor as ratio to entire width
    max_width_ratio : max width of censor as ratio to entire width
    min_height_ratio : minimum height of censor as ratio to entire height
    max_height_ratio : max height of censor as ratio to entire height

    Returns
    ------
    mask : a numpy ndarray of the size of the image, black mask with censored 
            parts outlined in white    
   
    Example
    -------
    redactometer.censor_dark('../img/2605160002.png', 0.05, 0.9, 0.005, 0.1)
    (above are approx good parameters)    

    """

    img = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #grayscale the image
    plt.gray()

    #im.shape = size of png, as np array
    mask = np.zeros(img.shape) #black mask in shape of image
    contours, hier = cv2.findContours(np.array(img), 
                                      cv2.RETR_LIST, 
                                      cv2.CHAIN_APPROX_SIMPLE)

    total_height, total_width = img.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, width, height = cv2.boundingRect(cnt)
        if (min_height_ratio * total_height < height < max_height_ratio * total_height) \
            and (min_width_ratio * total_width < width < max_width_ratio * total_width):
            cv2.drawContours(mask, [cnt], 0, 255, 1)

    plt.imshow(mask)
    plt.show()

    return mask
