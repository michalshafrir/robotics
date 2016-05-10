#!/usr/bin/env python

import rospy
import argparse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from tf import TransformListener
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_multiply
from tf.transformations import quaternion_inverse
from geometry_msgs.msg import Twist
import math
import cv2
from sensor_msgs.msg import CompressedImage
import numpy as np
from matplotlib import pyplot as plt
import threading
import copy
import rospkg


MIN_MATCH_COUNT = 10


#-------------------------------------------------------------------------------
# Object search class
#-------------------------------------------------------------------------------
class ObjectSearch:
  def __init__(self):

    # Navigation
    self.goalStatesText = [
                           'PENDING',
                           'ACTIVE',
                           'PREEMPTED',
                           'SUCCEEDED',
                           'ABORTED',
                           'REJECTED',
                           'PREEMPTING',
                           'RECALLING',
                           'RECALLED',
                           'LOST'
                          ]

    # Vision
    self.image = []
    self.processImage = False
    self.lock = threading.Lock()

    rospack = rospkg.RosPack()
    self.debugImageDir = rospack.get_path('assignment_5') + "/images/debug"
    self.trainImageDir = rospack.get_path('assignment_5') + "/images/train"
    self.trainImageNames = ['cereal', 'soup', 'pringles', 'kinect2', 'milk', 'straws', 'dressing']

    # Initialize node
    rospy.init_node('object_search')

    # Image subscriber and cv_bridge
    self.imageTopic = "/camera/rgb/image_raw/compressed"
    self.imageSub = rospy.Subscriber(self.imageTopic,CompressedImage,self.imageCallback)    
    self.debugImageId  = 0

    # Generate goal poses
    self.goalPoses = []
    self.goalPoses.append((  4.0,  2.6,  1.0))
    self.goalPoses.append((  2.0,  2.4, -3.0))
    self.goalPoses.append((  4.5,  0.7, -0.8))    

  def drawMatches(self, img1, kp1, img2, kp2, matches):

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    #print img2 == None
    #print img1 == None
    #cv2.imshow('img2',img2)
    #cv2.waitKey(0)

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = img1

    # Place the next image to the right of it
    out[:rows2,cols1:] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Also return the image if you'd like a copy
    return out

  #---------------------------
  # OpenCV Matching algorithm
  def match_img (self, img1, img2):
    # img1 - box
    # img2 - box in scene
    # Initiate SIFT detector
    sift = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #if(des1.type!=CV_32F):
    #  des1.convertTo(des1, CV_32F)
    

    #if(des1.type!=CV_32F):
    #  des2.convertTo(des2, CV_32F)
    

    matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)

       # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
       #            singlePointColor = None,
       #            matchesMask = matchesMask, # draw only inliers
       #            flags = 2)

        return (kp1,kp2,good);
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        return None;

  #-----------------------------------------------------------------------------
  # Image callback
  def imageCallback(self, data):

    # Capture image
    np_arr = np.fromstring(data.data, np.uint8)
    cv_img_color = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    cv_img = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # Store it if required
    self.lock.acquire()
    if self.processImage:
      print "Capturing image"
      self.image = copy.deepcopy(cv_img_color)
      self.processImage = False
      for target_name in self.trainImageNames:
        target_img = cv2.imread(self.trainImageDir + '/' + target_name+'.png',0)
        vals = self.match_img(target_img,cv_img)
        if (vals != None):
          (kp1,kp2,good) = vals
          print "Object found: "+target_name
          target_image = cv2.imread(self.trainImageDir + '/' + target_name+'.png',1)
          img3 = self.drawMatches( cv_img_color, kp1, target_img_color,kp2,good)
          plt.imshow(img3, 'gray'),plt.show() 
        else:
          print "Object NOT found: "+target_name 

    cv2.waitKey(1)

    self.lock.release() 

  #-----------------------------------------------------------------------------
  # Image callback
  def imageProcess(self, cv_img_name):

    cv_img = cv2.imread(cv_img_name,0)
    cv_img_color = cv2.imread(cv_img_name,1)    

    # Capture image
    #np_arr = np.fromstring(data.data, np.uint8)
    #cv_img_color = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    #cv_img = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # Store it if required
    self.lock.acquire()
    if self.processImage:
      print "Capturing image"
      self.image = copy.deepcopy(cv_img_color)
      self.processImage = False
      for target_name in self.trainImageNames:
        target_img = cv2.imread(self.trainImageDir + '/' + target_name+'.png',0)
        vals = self.match_img(target_img,cv_img)
        if (vals != None):
          (kp1,kp2,good) = vals
          print "Object found: "+target_name
          target_image = cv2.imread(self.trainImageDir + '/' + target_name+'.png',1)
          img3 = self.drawMatches( cv_img_color, kp1, target_img_color,kp2,good)
          plt.imshow(img3, 'gray'),plt.show() 
        else:
          print "Object NOT found: "+target_name 

    cv2.waitKey(1)

    self.lock.release() 


  #-------------------------------------------------------------------------------
  # Capture image, display it and save it to the debug folder
  def  capture_image(self):

    # First get the image to be processed
    self.processImage = True
    while (self.processImage):
      rospy.sleep(0.01)

    # Display image
    self.lock.acquire()
    cv2.imshow("Captured image", self.image)
    self.lock.release()

    # Save image
    cv2.imwrite(self.debugImageDir+"/image_" + str(self.debugImageId) + ".png", self.image)
    self.debugImageId += 1

    return self.debugImageDir+"/image_" + str(self.debugImageId) + ".png"


  #-----------------------------------------------------------------------------
  # Run!
  def run(self):

    # Just sit there doing nothing
    while True:
      img_name = self.capture_image()
      self.imageProcess(img_name)
      rospy.sleep(0.1)

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == '__main__':

  objectSearch = ObjectSearch()
  objectSearch.run()