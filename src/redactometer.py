import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import glob
import os, random

def unpaper(file, out):
    a = str(random.random())
    os.system("convert %s /tmp/%s.ppm"%(file, a))
    os.system("unpaper   --no-blurfilter --no-noisefilter  /tmp/%s.ppm /tmp/%s.out.ppm"%(a, a))
    os.system("convert /tmp/%s.out.ppm %s"%(a, out))


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
            #cv2.drawContours(orig_img, [cnt], 0, 255, 1) 
            censors.append(cnt)            

    #plt.imshow(mask)
    #plt.show()

    return mask, censors



def censor_fill(img_url, min_width_ratio=0.2, max_width_ratio=0.9,  min_height_ratio=0.2, max_height_ratio=0.9):
    #print min_width_ratio, max_width_ratio, min_height_ratio, max_height_ratio

    #print "img url", img_url
    img = cv2.imread(img_url)


    orig_img = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, 1)
    
    #grayscale the image
    plt.gray()


    #im.shape = size of png, as np array
    mask = np.zeros(img.shape) #black mask in shape of image
    # plt.imshow(img)
    # plt.show()

    contours, hier = cv2.findContours(np.array(img), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)


    #print len(contours)
    if len(contours) == 1: 
        #print cv2.boundingRect(contours[0])
        cv2.drawContours(img, [contours[0]], 0, 0, 100) 


    contours, hier = cv2.findContours(np.array(img), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    #print len(contours)
    total_height, total_width = img.shape

    censors = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    #print len(contours)
    for cnt in contours:        
        area = cv2.contourArea(cnt)
        x, y, width, height = cv2.boundingRect(cnt)
        if (min_height_ratio * total_height < height < max_height_ratio * total_height) \
            and (min_width_ratio * total_width < width < max_width_ratio * total_width):

            # and (min_width_ratio * total_width * min_height_ratio * total_height < area \
            #     < max_width_ratio * total_width * max_height_ratio * total_height):
            cv2.drawContours(mask, [cnt], 0, 255, -1) 
            cv2.drawContours(orig_img, [cnt], 0, colors[len(censors) % len(colors)], -1) 
            
            censors.append(cnt)            

    #plt.imshow(mask)
    #plt.show()

    return mask, censors, orig_img


def censor_dark_batch(source, destination, params):
    """Run the censor detection using given parameters on all images
    found in a given source dir. Dumps at destination dir. 
    Greyscale the image, and fill in the censors,
    and save resulting images. Make sure url ends in backslash"""
    
    imgs = glob.glob(source + '*') #glob ALL THE IMAGES
    for img in imgs:
        name = os.path.basename(img)
        mask, censors = censor_fill(img, **params)
        plt.imshow(mask)
        plt.savefig(destination + name)
        plt.close()
        print "saved figure : " + name

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

