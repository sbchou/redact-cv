import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import glob
import os, random
import match
from scipy import misc
import pandas

################  PRE PROCESSING ################ 
def deskew_dir(input_dir, output_dir):
    """Deskew all the images in a src directory and save in out dir"""
    for f in glob.glob(input_dir + '/*'):
        print "deskewing file ", f
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = match.deskew_text(img)
        deskewed = match.rotateImage(img, angle) 
        name = output_dir + os.path.basename(f)        
        misc.imsave(name, deskewed)

def sunpaper(file, out):
    a = str(random.random())
    os.system("convert %s /tmp/%s.ppm"%(file, a))
    os.system("unpaper -l single --pre-wipe --pre-border \
        --deskew-scan-deviation 2  /tmp/%s.ppm /tmp/%s.out.ppm"%(a, a))
    #os.system("unpaper   --no-blurfilter --no-noisefilter  /tmp/%s.ppm /tmp/%s.out.ppm"%(a, a))
    os.system("convert /tmp/%s.out.ppm %s"%(a, out))

def unpaper(input_dir, output_dir):
    """Unpaper all the files in input directory, store in output directory"""
    for f in glob.glob(input_dir + '/*'): 
        print "processing file ", f
        os.system("convert {file} {output_dir}{base}.ppm".format(\
            file=f, output_dir=output_dir, base=os.path.splitext(\
                os.path.basename(f))[0]))
        os.system("unpaper -l single --pre-wipe --pre-border \
            --deskew-scan-deviation 2 {output_dir}{base}.ppm {output_dir}{base}.out.ppm"\
            .format(input_dir=input_dir, output_dir=output_dir,\
             base=os.path.splitext(os.path.basename(f))[0]))
        os.system("convert {output_dir}{base}.out.ppm {output_dir}{base}.out.png"\
            .format(output_dir=output_dir, base=os.path.splitext(os.path.basename(f))[0]))
        os.system("rm {output_dir}*.ppm".format(output_dir=output_dir))
 
def draw_bounding(img_url, n):
    """Given input image, draw bounding rectangles on top of
    first n contours"""
    img = cv2.imread(img_url)    
    original = cv2.imread(img_url)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, 1)
    plt.gray()

    contours, hier = cv2.findContours(np.array(img), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1: 
        print cv2.boundingRect(contours[0])
        cv2.drawContours(img, [contours[0]], 0, 0, 100) 

    contours, hier = cv2.findContours(np.array(img), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    #sort contours by maximum area
    contours = sorted(contours, key=lambda cnt:cv2.contourArea(cnt), reverse=True)
     
    for cnt in contours[:n]:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(original, (x,y), (x+w,y+h), (0,255,0),2)

    plt.imshow(original)
    plt.show()

def blurred_detection(img_url, aperture):
    """Return the top left corner form of tuple (x, y) where x is xmin, y is ymin
    of the contours of n-largest areas, 
    after running image through a median blur filter with aperture (odd number)"""
    img = cv2.imread(img_url)

    original = cv2.imread(img_url)
    img = cv2.medianBlur(original, aperture)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, 1)

    plt.gray()

    contours, hier = cv2.findContours(np.array(img), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1: 
        #print cv2.boundingRect(contours[0])
        cv2.drawContours(img, [contours[0]], 0, 0, 100) 

    contours, hier = cv2.findContours(np.array(img), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    total_height, total_width = img.shape

    censors = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    contours = sorted(contours, key=lambda cnt:cv2.contourArea(cnt), reverse=True)

    #simply get average minimum area of censors after spotting and make that the cutoff
    for cnt in contours:    
        x, y, w, h = cv2.boundingRect(cnt)  
        area = cv2.contourArea(cnt)
        #if (100 < x < 750) and (100 < y < 870) \
        #    and area > 300:
        if 50 < y and area > 500:
            cv2.drawContours(original, [cnt], 0, colors[len(censors) % len(colors)], -1) 
            censors.append((x, y))            

    #plt.imshow(original)
    #plt.show()

    return censors, original 


def batch_blurred_detection(in_dir, out_dir, aperture):
    """Run the blurred detection on all images in a folder"""
    imgs = glob.glob(in_dir + '*') #glob ALL THE IMAGES

    results = []

    for img in imgs:
        name = os.path.basename(img)
        censors, original = blurred_detection(img, aperture)

        plt.imshow(original)
        plt.savefig(out_dir + name)
        plt.close()
        
        for censor in censors:
            results.append((name, censor[0], censor[1]))
    
    results = pandas.DataFrame(results, columns=["img", "x", "y"])
    return(results)

def plot_comp(img1, img2):
    """Plot two images side by side"""
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(img1)
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(img2)
    plt.show()

#### Initial Paramters #############################################
def censor_fill(img_url, min_width_ratio=0.2, max_width_ratio=0.9,  \
    min_height_ratio=0.2, max_height_ratio=0.9, offset_x=0.05, offset_y=0.10):
    #print min_width_ratio, max_width_ratio, min_height_ratio, max_height_ratio

    #print "img url", img_url
    img = cv2.imread(img_url)

    original = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #threshold assigns pixel > value to color
    _, img = cv2.threshold(img, 127, 255, 1)
    
    #grayscale the image
    plt.gray()

    #lines = cv2.HoughLinesP(img4, 1, math.pi / 4.0, 100, None, 100, 10)
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


        #the x, y = left-top corner of boundingrectangle, aka xmin, ymin
        if offset_x * total_width < x < total_width - offset_x * total_width - width \
            and offset_y * total_height < y < total_height - offset_y * total_height - height \
            and (min_height_ratio * total_height < height < max_height_ratio * total_height) \
            and (min_width_ratio * total_width < width < max_width_ratio * total_width):
            
            #compute pixel points
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            pixelpoints = np.transpose(np.nonzero(mask))
            #import pdb; pdb.set_trace()
            mean_val = cv2.mean(img, mask=mask)
            print "mean", mean_val


            rect_area = width * height
            extent = float(area)/rect_area

            if extent > 0.2 and mean_val > 200:

                #compute solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull) 
                solidity = float(area)/hull_area

                cv2.drawContours(original, [cnt], 0, colors[len(censors) % len(colors)], -1) 
                censors.append(cnt)            

    return censors, original

def batch_censor_dark(source, destination, params):
    """Run the censor detection using given parameters on all images
    found in a given source dir. Dumps at destination dir. 
    Greyscale the image, and fill in the censors,
    and save resulting images. Make sure url ends in backslash"""
    results = []
    imgs = glob.glob(source + '*') #glob ALL THE IMAGES
    for img in imgs:
        name = os.path.basename(img)
        censors, original = censor_fill(img, **params)
        results.append((name, len(censors)))
        plt.imshow(original)
        plt.savefig(destination + name)
        plt.close()
        print "saved figure : " + name
    return results




#### Future work ################################################
def template_match(img_url, template_url, outfile_name, threshold):
    """
    For input img found in img_url, and template image found at
    template_url, outline all matches and output in new file.
    """

    img_rgb = cv2.imread(img_url)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)()
    template = cv2.imread(template_url, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite(outfile_name, img_rgb)


