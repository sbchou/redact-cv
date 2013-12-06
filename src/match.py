"Code for converting images to layouts."

import math
import matplotlib.pyplot as plt
import cv2, sys
import numpy as np
import numpy.linalg
import scipy.ndimage as ndimage
from random import random
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

PARAMS = {
  "feature_score": 5000
}

flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)

# Debug parameter.
NO_SHOW = True

class Features:
  "Object for feature computation."
  def __init__(self):
    surf = cv2.SURF(500, 4, 2, False, True) #cv2.FeatureDetector_create("SURF")
    self.detect = surf
    self.extract = cv2.DescriptorExtractor_create("SURF")

    
  def compute_features(self, img):
    key1, des1 = self.detect.detectAndCompute(img, None)
    return key1, des1  

def show_image(img, half = False, override = False):
  "Show an image (if NO_SHOW is false)."
  if not override and NO_SHOW: return 
  cv2.namedWindow('Display Window')        ## create window for display
  if half:
    newimg = cv2.resize(img, (int(img.shape[1] / 2.0), int(img.shape[0] / 2.0)))
  else:
    newimg = img
  cv2.imshow('Display Window', newimg)         ## Show image in the window
  print "size of image: ",img.shape        ## print size of image
  cv2.waitKey(0)                           ## Wait for keystroke
  cv2.destroyAllWindows()

  plt.gray()
  plt.imshow(newimg)

def show_pairs(img1, warpImage, pairs, key1, key2, override = False):
  "Helper function, show a pair of images."
  img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
  warpImage = cv2.cvtColor(warpImage, cv2.COLOR_GRAY2BGR)
  h1, w1 = img.shape[:2]
  h2, w2 = warpImage.shape[:2]
  nWidth = w1+w2
  nHeight = max(h1, h2)
  hdif = (h1-h2)/2
  newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
  newimg[:h1, :w1] = img
  newimg[:h2, w1:w1+w2] = warpImage
  show_image(newimg, half=False, override = override)

  for p1, p2 in pairs:
     x1, y1 = key1[p1].pt
     x2, y2 = key2[p2].pt
     pt_a = (int(x1), int(y1+hdif))
     pt_b = (int(x2 + w1), int(y2))
     cv2.rectangle(newimg, (pt_a[0] + 5, pt_a[1] + 5), 
                   (pt_a[0] - 5, pt_a[1] - 5), (255,0,0))
     cv2.rectangle(newimg, (pt_b[0] + 5, pt_b[1] + 5), 
                   (pt_b[0] - 5, pt_b[1] - 5), (255,0,0))
  show_image(newimg, half = True)
  return newimg

def remove_blobs(img, max_size = 1000, min_height = 5):
  "Remove blobs above a fixed size from the image."
  ret, img2 = cv2.threshold(img, 250, 255, 1)
  contours, hier = cv2.findContours(np.array(img2), 
                                    cv2.RETR_LIST, 
                                    cv2.CHAIN_APPROX_SIMPLE)
  mask = np.zeros(img2.shape, np.uint8)
  for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, width, height = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > max_size or height < min_height:
      cv2.drawContours(mask,[cnt],0, 255, -1)
  show_image(mask)
  ret = cv2.bitwise_and(img2, cv2.bitwise_not(mask))
  return ret

def match_flann(desc1, desc2, key1, key2, max_dist = 0.05):
  "Nearest neighbors matching of sets of features."
  flann = cv2.flann_Index(desc2, flann_params)
  idx2, dist = flann.knnSearch(desc1, 5, params = {}) # bug: need to provide empty dict
  matches = [-1] * len(desc1)
  for id1, match2 in zip(range(len(desc1)), idx2):
    best = 100
    for i, m in enumerate(match2):
      if dist[id1][i] > max_dist: continue
      diff = abs(np.array(key1[id1].pt) - numpy.array(key2[m].pt))
      l2 = numpy.linalg.norm(diff)
      if l2 < best:
        best = l2 
        matches[id1] = m
        break
  return [(i,m) for i, m in enumerate(matches) if m != -1] 

def angle(p1, p2):
  "Compute the angle frow two points."
  xDiff = p2[0] - p1[0];
  yDiff = p2[1] - p1[1];
  return math.atan2(yDiff, xDiff) * (180 / math.pi)

def warped_image_from_match(img1, img2, key1, key2, pairs, mask):
  "Warp the image based on RANSAC matching."
  p1 = np.array([key1[p1].pt for p1, _ in pairs])
  p2 = np.array([key2[p2].pt for _, p2 in pairs])

  val, trans = cv2.findHomography(p2, p1, cv2.RANSAC)
  return (cv2.warpPerspective(img2, val, (img1.shape[1], img1.shape[0])), \
      cv2.warpPerspective(mask, val, (img1.shape[1], img1.shape[0])))

def rotateImage(image, angle):
  "Rotate the image by an angle"
  image_center = tuple(np.array(image.shape) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle,1.0)
  result = cv2.warpAffine(image, rot_mat, 
                          (image.shape[1], image.shape[0]), 
                          flags=cv2.INTER_LINEAR)
  return result


def compute_final(warp, img):
  "Run the processing pipeline on the image."
  warp2 = remove_blobs(warp[100:-100, 100: -100])
  img2 = remove_blobs(img[100:-100, 100: -100])

  angle = deskew_text(img2)
  angle2 = deskew_text(warp2)
  average_angle = (angle + angle2) / 2.0

  mask = np.zeros(img2.shape, np.uint8)
  boxes1 = count_lines(rotateImage(img2, average_angle), mask)

  mask2 = np.zeros(img2.shape, np.uint8)
  boxes2 = count_lines(rotateImage(warp2, average_angle), mask2)

  d= np.array(cv2.absdiff(rotateImage(img2, average_angle), rotateImage(warp2, average_angle)))
  show_pairs(rotateImage(img2, average_angle), rotateImage(warp2, average_angle), [], [], [])
  return boxes2, boxes1

def deskew_text(img2):  
  "Rotate the image so text is straigt."
  img5 = ndimage.uniform_filter(img2, (1, 50))
  show_image(img5)

  ret, img5 = cv2.threshold(img5, 50, 255, cv2.THRESH_BINARY)
  lines = cv2.HoughLinesP(img5, 1, math.pi / 180.0, 100, None, 100, 20)
  cum_angle = 0
  for l in lines[0]:
    a = angle((l[0], l[1]), (l[2], l[3]))
    if abs(a) < 5:
      cum_angle += a
  rotate = cum_angle / len(lines[0])
  return rotate

def count_lines(img2, mask2):
  "Compute the layout from the image"
  draw_image = numpy.array(img2)
  img4 = numpy.array(img2)
  ret, img4 = cv2.threshold(img4, 4, 255, cv2.THRESH_BINARY)

  # TODO - Clean this up.
  #img4 = ndimage.uniform_filter(img4, (1, 10))
  img4 = ndimage.uniform_filter(img4, (1, 50))
  show_image(img4)
  # img4 = cv2.adaptiveThreshold(img4, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 0) 
  #cv2.Canny(img4, 1, 1) #
  ret, img4 = cv2.threshold(img4, 50, 255, cv2.THRESH_BINARY)
  show_image(img4)
  # img4 = cv2.Canny(img4, 1, 1) #cv2.adaptiveThreshold(img4, 1.0, 1)

  # show_image(img4)
  lines = cv2.HoughLinesP(img4, 1, math.pi / 4.0, 100, None, 100, 10)
  # show_image(img4)
  mask = np.zeros(img4.shape, np.uint8)
  cum_angle = 0
  for l in lines[0]:
    a = angle((l[0], l[1]), (l[2], l[3]))
    if abs(a) < 1.0: 
      cv2.line(mask, (l[0], l[1]), (l[2], l[3]), 255, 1)
    cum_angle += a
  show_image(mask)
  contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_TC89_L1)  
  boxes = []
  draw_image = cv2.cvtColor(draw_image, cv2.COLOR_GRAY2BGR)
  for cnt in contours:
    box  = cv2.boundingRect(cnt)
    x, y, width, height = box
    if width > 20 and height > 2 and height < 40:
      cv2.rectangle(mask2, (x, y ), (x + width , y + height ), (250, 0, 0), 3)
      cv2.rectangle(draw_image, (x + 5, y - 5), (x + width + 10, y + height + 10), (250, 0, 0), 3)
      boxes.append((x, y, width , height))
  plt.imshow(mask2)
  # plt.imshow(draw_image)
  # plt.savefig("/tmp/outline.png")
  show_image(draw_image)
  return boxes

def process_pair(img1, img2, im1_mask, im2_mask):
  "Convert a pair of images into layouts."
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  feats = Features()
  key1, des1 = feats.compute_features(img1)
  key2, des2 = feats.compute_features(img2)
  pairs = match_flann(des1, des2, key1, key2)

  if len(pairs) < 10: 
    # Failed to find a valid match
    return
  show_pairs(img1, img2, pairs, key1, key2)
  warped, warped_mask = \
      warped_image_from_match(img1, img2, key1, key2, pairs, im2_mask)
  show_image(cv2.absdiff(remove_blobs(img1), remove_blobs(warped)))
  b1, b2 = compute_final(img1, warped)

  return b1, b2, img1, warped, im1_mask, warped_mask

if __name__ == "__main__":
  im1, im2 = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
  
  im1_mask = np.zeros(im1.shape[:2], dtype = np.uint8)
  im2_mask = np.zeros(im2.shape[:2], dtype = np.uint8)

  process_pair(im1, im2, im1_mask, im2_mask)
