import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def censor_dark(img_url, min_width_ratio=0.2, max_width_ratio=0.9,  min_height_ratio=0.2, max_height_ratio=0.9):
    #print min_width_ratio, max_width_ratio, min_height_ratio, max_height_ratio

    #print "img url", img_url
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

    censors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, width, height = cv2.boundingRect(cnt)
        if (min_height_ratio * total_height < height < max_height_ratio * total_height) \
            and (min_width_ratio * total_width < width < max_width_ratio * total_width):
            cv2.drawContours(mask, [cnt], 0, 255, 1) 
            censors.append(cnt)            

    #plt.imshow(mask)
    #plt.show()

    return mask, censors



def censor_test(img_url, min_width_ratio=0.2, max_width_ratio=0.9,  min_height_ratio=0.2, max_height_ratio=0.9):
    #print min_width_ratio, max_width_ratio, min_height_ratio, max_height_ratio

    #print "img url", img_url
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

    censors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, width, height = cv2.boundingRect(cnt)
        if (min_height_ratio * total_height < height < max_height_ratio * total_height) \
            and (min_width_ratio * total_width < width < max_width_ratio * total_width):
            #cv2.drawContours(mask, [cnt], 0, 255, -1) 
            censors.append(cnt)            

    plt.imshow(mask)
    plt.show()

    return mask, censors


def template_match(img_url, template_url, outfile_name, threshold):
    """
    For input img found in img_url, and template image found at
    template_url, outline all matches and output in new file.
    """

    img_rgb = cv2.imread(img_url)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_url, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite(outfile_name, img_rgb)

