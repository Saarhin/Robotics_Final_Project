#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import numpy as np

import cv2 as cv
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info = f"/{self._vehicle_name}/camera_node/camera_info"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        rospy.loginfo("Camera parameters finding...")
        # construct subscriber
        self.sub_info = rospy.Subscriber(self._camera_info, CameraInfo, self.callback_info)
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback_image)

        self.K = None
        self.D = None
        self._custom_topic = f"/{self._vehicle_name}/custom_node/image/compressed"
        self.pub = rospy.Publisher(self._custom_topic, Image) # queue_size=10
        self.disorted_image = None

    def callback_image(self, msg):
        # convert JPEG bytes to CV image
        rate = rospy.Rate(3)
        if self.K is None:
            return
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        h,w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.K, self.D, (w,h), 1, (w,h))
        dst = cv.undistort(image, self.K, self.D, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        self.disorted_image = dst

    def callback_info(self, msg):
        rate = rospy.Rate(1)
        # https://stackoverflow.com/questions/55781120/subscribe-ros-image-and-camerainfo-sensor-msgs-format
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        # https://github.com/IntelRealSense/realsense-ros/issues/709ss
        self.K = np.array(msg.K).reshape(3, 3)
        self.D = np.array(msg.D)
        # rospy.loginfo("Camera parameters received.")
        rate.sleep()
    
    def start(self):
        # https://stackoverflow.com/questions/55377442/how-to-subscribe-and-publish-images-in-ros
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():       
            if self.disorted_image is not None:
                # rospy.loginfo('publishing image')
                image_msg = self._bridge.cv2_to_imgmsg(self.disorted_image, encoding="bgr8")
                self.pub.publish(image_msg)
            #self.pub.publish(self.raw_image)
            rate.sleep()

    def color_detect_start(self):
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():       
            if self.disorted_image is not None:
                self.color_detect()
                # rospy.loginfo('publishing image')
                image_msg = self._bridge.cv2_to_imgmsg(self.color_detect_image, encoding="bgr8")
                self.pub.publish(image_msg)
            #self.pub.publish(self.raw_image)
            rate.sleep()
        
    def image_preprocess(self):
        new_width = 400
        new_height = 300
        resized_image = cv.resize(self.disorted_image, (new_width, new_height), interpolation = cv.INTER_AREA)
        blurred_image = cv.blur(resized_image, (5, 5)) 
        return blurred_image
    
    def color_detect(self):
        # rate = rospy.Rate(3)
        if self.K is None:
            return
        imageFrame = self.image_preprocess().astype(np.uint8)
        hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV)

        # Set range for red color and 
        # define mask 
        red_lower = np.array([110, 80, 70], np.uint8) 
        red_upper = np.array([120, 250, 240], np.uint8) 
        red_mask = cv.inRange(hsvFrame, red_lower, red_upper) 

        # Set range for green color and 
        # define mask 
        green_lower = np.array([36, 50, 70], np.uint8) 
        green_upper = np.array([89, 255, 255], np.uint8) 
        green_mask = cv.inRange(hsvFrame, green_lower, green_upper) 

        # Set range for blue color and 
        # define mask 
        blue_lower = np.array([100, 50, 70], np.uint8) 
        blue_upper = np.array([120, 255, 255], np.uint8) 
        blue_mask = cv.inRange(hsvFrame, blue_lower, blue_upper) 

        # Morphological Transform, Dilation 
        # for each color and bitwise_and operator 
        # between imageFrame and mask determines 
        # to detect only that particular color 
        kernel = np.ones((5, 5), "uint8") 
        
        # For red color 
        red_mask = cv.dilate(red_mask, kernel) 
        res_red = cv.bitwise_and(imageFrame, imageFrame, 
                                mask = red_mask) 
        
        # For green color 
        green_mask = cv.dilate(green_mask, kernel) 
        res_green = cv.bitwise_and(imageFrame, imageFrame, 
                                    mask = green_mask) 
        
        # For blue color 
        blue_mask = cv.dilate(blue_mask, kernel) 
        res_blue = cv.bitwise_and(imageFrame, imageFrame, 
                                mask = blue_mask) 

        # Creating contour to track red color 
        contours, hierarchy = cv.findContours(red_mask, 
                                            cv.RETR_TREE, 
                                            cv.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv.boundingRect(contour) 
                imageFrame = cv.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2) 
                
                cv.putText(imageFrame, "Blue Colour", (x, y), 
                            cv.FONT_HERSHEY_SIMPLEX, 1.0, 
                            (0, 0, 255)) 
        # Creating contour to track green color 
        contours, hierarchy = cv.findContours(green_mask, 
                                            cv.RETR_TREE, 
                                            cv.CHAIN_APPROX_SIMPLE) 
        self.color_detect_image = imageFrame   
    

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    node.color_detect_start()
    # keep spinning
    rospy.spin()