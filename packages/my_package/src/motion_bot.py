#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from duckietown_msgs.msg import Twist2DStamped
import cv2 
from cv_bridge import CvBridge
import numpy as np
from duckietown_msgs.msg import LEDPattern 
import dt_apriltags as aptag

from std_msgs.msg import Header, ColorRGBA, Int32, String
import math

class MotionBotNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MotionBotNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info = f"/{self._vehicle_name}/camera_node/camera_info"

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.twisted_publisher = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        
        self._v = 0.5
        self._omega = 0
        
        
        self._bridge = CvBridge()

        self.sub_info = rospy.Subscriber(self._camera_info, CameraInfo, self.callback_info)
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback_image)

        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)

        self.K = None
        self.D = None

        self.undisorted_image = None
        self.redline_image = None
        self.leader_duckiebot_image = None
        self.leader_duckiebot_turn = None
        self.apriltag_image = None
        self.crosswalk_image = None
        self.gray = None

        self.control_type = "PID"
        self.proportional_gain = 0.05
        self.derivative_gain = 0.03
        self.integral_gain = 0.001

        #color detection
        self.white_lower = np.array([0, 0, 180], np.uint8) 
        self.white_upper = np.array([180, 40, 255], np.uint8)

        self.yellow_lower = np.array([20, 80, 100], np.uint8) 
        self.yellow_upper = np.array([40, 255, 255], np.uint8) 

        self.error = 0
        self.prev_error = 0
        self.history = np.zeros((1,10))
        # self.history_leader_duckiebot = np.zeros((1,30), dtype=float)
        self.integral = 0
        self.calibration = -90

        self.rate = rospy.Rate(3)

        self.timer_avoid_redline = 30
        self.timer_stop = 10
        self.counter_avoid_red = 0
        self.counter_stop = 0

        self.prev_x = (1.0, 1.0, 1.0, 1.0)
        self.x = (1.0, 1.0, 1.0, 1.0)

        self.mode = 0

        # mode = 0  ->  pid_control
        # mode = 1  ->  stop for red line
        # mode = 2  ->  pid avoid stoping before the red line
        # mode = 3  ->  Sees the leader Duckiebot

        self.count_stops = 0
        self.predict_turn = "straight"

        self.left_turn_dist = 2
        self.right_turn_dist = 0.5
        self.straight_turn_dist = 2
        self.when_to_detect_tag = 0

        self.tag_id = 0

        self.stop_before_crosswalk = 0
        self.timer_avoid_crosswalk = 60
        self.timer_stop_crosswalk = 10

        self.counter_yellow_line = 0
        self.timer_yellow_line = 50


        #test
        self._custom_topic_lane = f"/{self._vehicle_name}/custom_node/image/black"
        self.pub_lane = rospy.Publisher(self._custom_topic_lane, Image, queue_size=1)


    def callback_info(self, msg):

        # https://stackoverflow.com/questions/55781120/subscribe-ros-image-and-camerainfo-sensor-msgs-format
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        # https://github.com/IntelRealSense/realsense-ros/issues/709ss
        self.K = np.array(msg.K).reshape(3, 3)
        self.D = np.array(msg.D)

    def callback_image(self, msg):

        if self.K is None:
            return
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        h,w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w,h), 1, (w,h))
        dst = cv2.undistort(image, self.K, self.D, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        self.undisorted_image = self.image_preprocess(dst)
        self.redline_image = self.redline_image_process(self.undisorted_image)
        self.leader_duckiebot_image = self.leader_duckiebot_image_process(self.undisorted_image)
        self.leader_duckiebot_turn = self.leader_duckiebot_turn_process(self.undisorted_image)
        self.crosswalk_image = self.crosswalk_image_process(self.undisorted_image)
        self.apriltag_image = self.apriltag_image_process(self.undisorted_image)
        self.gray = self.calc_error(self.undisorted_image)
        # image_msg = self._bridge.cv2_to_imgmsg(self.crosswalk_image, encoding="rgb8")
        # self.pub_lane.publish(image_msg)

    def redline_image_process(self, img):
        h, w, _ = img.shape
        resized_image = img[h//2:, w//4: -w//4, :]
        return resized_image
    
    def leader_duckiebot_image_process(self, img):
        h, w, _ = img.shape
        resized_image = img[: , w//4:-w//4, :]
        return resized_image
    
    def leader_duckiebot_turn_process(self, img):
        h, w, _ = img.shape
        resized_image = img[:-h//4 , :, :]
        return resized_image
    
    def apriltag_image_process(self, img):
        h, w, _ = img.shape
        resized_image = img[: , w//4:, :]

        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        return image

    def crosswalk_image_process(self, img):
        h, w, _ = img.shape
        resized_image = img[h//4: , w//5:-w//5, :]
        return resized_image

    def image_preprocess(self, img):

        new_width = 400
        new_height = 300
        resized_image = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
        blurred_image = cv2.blur(resized_image, (5, 5)) 
        return blurred_image
    
    def calc_error(self, imageFrame):

        height = imageFrame.shape[0]
        imageFrame = imageFrame[height//2:-height//5, :, :]


        imageFrame = cv2.GaussianBlur(imageFrame, (5, 5), 0)

        kernel = np.ones((5, 5), "uint8") 

        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        if self.mode == 13:
            yellow_mask = cv2.inRange(hsvFrame, self.yellow_lower, self.yellow_upper) 

            yellow_mask = cv2.dilate(yellow_mask, kernel) 
            res_yellow = cv2.bitwise_and(imageFrame, imageFrame, 
                                    mask = yellow_mask) 

            lane_mask = np.zeros_like(yellow_mask)  
            lane_mask[yellow_mask > 0] = 255 

        else:

            white_mask = cv2.inRange(hsvFrame, self.white_lower, self.white_upper) 

            # For white color 
            white_mask = cv2.dilate(white_mask, kernel) 
            res_white = cv2.bitwise_and(imageFrame, imageFrame, 
                                    mask = white_mask) 

            lane_mask = np.zeros_like(white_mask)  
            lane_mask[white_mask > 0] = 255 


        contours, hierarchy = cv2.findContours(lane_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        if len(contours)>0:
            # Sort contours by area in descending order and pick the top two
            max_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

            x_values = max_contour[:, 0, 0]  # Extracting x-coordinates

            # Compute the average x-coordinate
            avg_x = np.mean(x_values)
        else:
            avg_x = 0


        self.error = avg_x- lane_mask.shape[1]/2.0 + self.calibration

        return lane_mask
        
    def publish_twisted(self, v, omega):

        message = Twist2DStamped(v=v, omega=omega)
        self.twisted_publisher.publish(message)

    def pid_controller(self):
        self.history = np.roll(self.history, shift=-1, axis=1)  # Shift all values left
        self.history[0, -1] = self.error
        self.integral = np.sum(self.history)
        derivative = self.error - self.prev_error 
        self.prev_error = self.error
        return self.proportional_gain * self.error + self.derivative_gain * derivative + self.integral_gain * self.integral

    def move_pid(self):
            
        control = self.pid_controller()
        self.publish_twisted(v=self._v, omega = -1*control)
        self.calc_error(self.undisorted_image)
        

    def stop(self):
        self.publish_twisted(v = 0, omega = 0)

    def on_shutdown(self):
        self.publish_twisted(v = 0, omega = 0)
        self.publish_leds((1.0, 1.0, 1.0, 1.0))

    def detect_red_line(self, image):
        if image is None:
            return
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red_ranges = {'lower': np.array([0, 150, 50]), 'upper': np.array([10, 255, 255])}
        

        mask = cv2.inRange(hsv, red_ranges['lower'], red_ranges['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Estimate distance based on contour position
                image_height = image.shape[0]
                distance = (image_height - (y + h)) / image_height
                
                return True, distance
                    
        return False, float("inf")
    
    def detect_leader_duckiebot(self, image):
        if image is None:
            return
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        black_ranges = {'lower': np.array([110, 150, 100]), 'upper': np.array([120, 250, 200])}
        

        mask = cv2.inRange(hsv, black_ranges['lower'], black_ranges['upper'])

       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 2:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Estimate distance based on contour position
                image_height = image.shape[0]
                distance = (image_height - (y + h)) / image_height
                middle = x+w//2
                if middle < image.shape[1]//2 and abs(middle - image.shape[1]//2) > 30:
                    return True , distance, "left"
                if middle > image.shape[1]//2 and abs(middle - image.shape[1]//2) > 30:
                    return True , distance, "right"
                return True, distance,  "straight"
            
        return False, float("inf"), "straight"
    
    def detect_leader_duckiebot_turn(self, image):
        if image is None:
            return
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        black_ranges = {'lower': np.array([110, 80, 50]), 'upper': np.array([130, 255, 255])}
        

        mask = cv2.inRange(hsv, black_ranges['lower'], black_ranges['upper'])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 2:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Estimate distance based on contour position
                image_height = image.shape[0]
                distance = (image_height - (y + h)) / image_height
                middle = x+w//2
                if middle < image.shape[1]//2 and abs(middle - image.shape[1]//2) > 50:
                    return True , distance, "left"
                if middle > image.shape[1]//2 and abs(middle - image.shape[1]//2) > 50:
                    return True , distance, "right"
                return True, distance,  "straight"
            
        return False, float("inf"), "straight"
            
    
    def detect_tag(self):
        detector = aptag.Detector(families="tag36h11")
        results = detector.detect(self.apriltag_image)

        while not results:
            results = detector.detect(self.apriltag_image)

        
       
           
        def area(r):
            # Use corners to compute polygon area
            (ptA, ptB, ptC, ptD) = r.corners
            return 0.5 * abs(
                ptA[0]*ptB[1] + ptB[0]*ptC[1] + ptC[0]*ptD[1] + ptD[0]*ptA[1]
                - ptB[0]*ptA[1] - ptC[0]*ptB[1] - ptD[0]*ptC[1] - ptA[0]*ptD[1]
            )

        largest_tag = max(results, key=area)

        tag_id = str(largest_tag.tag_id)
        return tag_id
    
    def detect_crosswalk(self, image):
        if image is None:
            return
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        black_ranges = {'lower': np.array([110, 150, 100]), 'upper': np.array([120, 250, 200])}
        

        mask = cv2.inRange(hsv, black_ranges['lower'], black_ranges['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_msg = self._bridge.cv2_to_imgmsg(mask, encoding="8UC1")
        self.pub_lane.publish(image_msg)
        
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 400:
                
                return True
                    
        return False
    
    def publish_leds(self, x):      
        if self.gray is not None:
            msg = LEDPattern()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            color_msg = ColorRGBA()
            color_msg.r, color_msg.g, color_msg.b, color_msg.a = x


            # Set LED colors
            msg.rgb_vals = [color_msg] * 5
            self.led_pub.publish(msg) 
        pass

    def detect_ducks(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        duck_ranges = {'lower': np.array([9, 91, 163]), 'upper': np.array([22, 255, 255])}
    
        mask = cv2.inRange(hsv, duck_ranges['lower'], duck_ranges['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.contourArea(largest_contour) > 500
        
        return False
                  
    
    def turn_left(self):
        distance_traveled = 0
        dt = 0.1

        while distance_traveled < self.left_turn_dist:
            self.publish_twisted(v=self._v, omega = 1)
            self.calc_error(self.undisorted_image)
            self.rate.sleep()
            distance_traveled += self._v * dt 

        self.predict_turn = "straight"
        rospy.loginfo("done with the left turn")

    def go_straight(self):
        distance_traveled = 0
        dt = 0.1

        while distance_traveled < self.straight_turn_dist:
            self.publish_twisted(v=self._v, omega = 0)
            self.calc_error(self.undisorted_image)
            self.rate.sleep()
            distance_traveled += self._v * dt 

        self.predict_turn = "straight"
        rospy.loginfo("done with the moving forward")

    def turn_right(self):
        distance_traveled = 0
        dt = 0.1

        while distance_traveled < self.right_turn_dist:
            self.publish_twisted(v=self._v, omega = -2.5)
            self.calc_error(self.undisorted_image)
            self.rate.sleep()
            distance_traveled += self._v * dt 

        self.predict_turn = "straight"  
        rospy.loginfo("done with the right turn")
        
    def detect_broken_duckiebot(self, image):
        if image is None:
            return
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        black_ranges = {'lower': np.array([110, 150, 100]), 'upper': np.array([120, 250, 200])}
        

        mask = cv2.inRange(hsv, black_ranges['lower'], black_ranges['upper'])

       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 2 and cv2.contourArea(largest_contour) < 300:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Estimate distance based on contour position
                image_height = image.shape[0]
                distance = (image_height - (y + h)) / image_height
                
                return True, distance
            
        return False, float("inf")

    def run(self):
        
        self.rate.sleep()
        if self.mode < 9:
            leader_see, leader_distance, temp_dir = self.detect_leader_duckiebot(self.leader_duckiebot_image)
            if leader_see:
                self.predict_turn = temp_dir
        else:
            leader_see = False
            leader_distance = float("inf")

       

        # checking if we should change the mode based on the info we are getting
        # if self.mode == 14:
        #     self.mode = 9
        # if self.mode == 13 and self.counter_yellow_line == self.timer_yellow_line:
        #     self.mode = 14


        # if self.mode == 13 and self.counter_yellow_line < self.timer_yellow_line:
        #     self.counter_yellow_line += 1

        # if self.mode == 12 and self.detect_broken_duckiebot(self.leader_duckiebot_image)[1] < 0.7:
        #     self.mode = 13
        #     self.counter_yellow_line = 0

        # if self.mode == 11 and self.stop_before_crosswalk < self.timer_avoid_crosswalk:
        #     self.stop_before_crosswalk +=1
        
        # if self.mode == 11 and self.stop_before_crosswalk == self.timer_avoid_crosswalk:
        #     self.mode = 12
        #     self.stop_before_crosswalk = 0
        
        # if self.mode == 10: 
        #     if  self.stop_before_crosswalk < self.timer_stop_crosswalk and not self.detect_ducks(self.crosswalk_image):
        #         self.stop_before_crosswalk +=1

        # if self.mode == 10 and self.stop_before_crosswalk == self.timer_stop_crosswalk:
        #     self.mode = 11
        #     self.stop_before_crosswalk = 0
        
        # if self.mode == 9 and self.detect_crosswalk(self.crosswalk_image):
        #     self.mode = 10
        #     self.stop_before_crosswalk = 0

           
        # if self.mode == 8:
        #     self.mode = 9

        # if self.mode == 5 or self.mode == 6 or self.mode == 7:
        #     self.mode = 0

        # if self.mode == 4 and not leader_see:
        #     self.mode = 0
            
        # if self.mode == 4 and leader_distance >= 0.6:
        #     self.mode = 3

        # if leader_see and leader_distance >= 0.6:
        #     self.mode = 3

        # if leader_see and leader_distance < 0.6:
        #     self.mode = 4

        # if self.mode == 1 and self.counter_stop < self.timer_stop: # 1 -> 1 stop for some time before the red line
        #     self.counter_stop += 1
        #     see, _, temp_dir= self.detect_leader_duckiebot_turn(self.leader_duckiebot_turn)
        #     rospy.loginfo("done with waiting for red line")
        #     if see:
        #         self.predict_turn = temp_dir
            

        # if self.mode == 1 and self.counter_stop == self.timer_stop: # 1 -> 2 start moving without detecting the red line  
        #     rospy.loginfo(self.predict_turn )
        #     self.when_to_detect_tag += 1
        #     if self.when_to_detect_tag < 4 :
        #         if self.predict_turn == "straight":
        #             self.mode = 7
        #             rospy.loginfo("straight")
        #         elif self.predict_turn == "left":
        #             self.mode = 5
        #             rospy.loginfo("left")
        #         elif self.predict_turn == "right":
        #             self.mode = 6
        #             rospy.loginfo("right")

        #     self.counter_stop = 0
            
        
        # if self.mode == 1 and  (self.when_to_detect_tag == 4 or self.when_to_detect_tag == 5):
        #     self.tag_id = self.detect_tag()
        #     self.mode = 8
            

        # if (self.mode == 0 or self.mode == 3 or self.mode == 4 or self.mode == 9) and self.detect_red_line(self.redline_image)[0]: #  0 -> 1 if detect_lane = true
        #     self.mode = 1
        #     self.count_stops = 0


        # if (self.mode == 2 or self.mode == 4 or self.mode == 5) and self.counter_avoid_red < self.timer_avoid_redline: # 2 -> 2 still don't want to detect the red line
        #     self.counter_avoid_red +=1

        # if self.mode == 2 and self.counter_avoid_red == self.timer_avoid_redline: # 2 - > 0 you can check if you see red line
        #     self.counter_avoid_red = 0
        #     self.mode = 0
        
        
        
        # deciding what to do based on the mode we are in
        
        if self.mode == 0 or self.mode == 2:
            self.move_pid()
            self.x = (1.0, 1.0, 1.0, 1.0) # white

        if self.mode == 1: 
            self.stop()
            self.x = (1.0, 1.0, 1.0, 1.0) # white
        
        if self.mode == 3:
            self.x = (0.0, 0.0, 1.0, 1.0) # green
            self.move_pid()

        if self.mode == 4:
            self.x = (1.0, 0.0, 1.0, 1.0) # green
            self.stop()

        if self.mode == 5:
            self.x = (1.0, 1.0, 1.0, 1.0) # white
            self.turn_left()
            
        if self.mode == 6:
            self.x = (1.0, 1.0, 1.0, 1.0) # white
            self.turn_right()

        if self.mode == 7:
            self.x = (1.0, 1.0, 1.0, 1.0) # white
            self.go_straight()

        if self.mode == 8:
            if int(self.tag_id) == 48:
                self.turn_right()
            elif int(self.tag_id) == 50:
                self.turn_left()

        if self.mode == 9 or self.mode == 11 or self.mode == 12:
            self.move_pid()
            self.x = (1.0, 1.0, 1.0, 1.0) # white

        if self.mode == 10:
            self.stop()
            self.x = (1.0, 1.0, 0.0, 1.0) # white

        if self.mode == 13:
            self.move_pid()
            self.x = (1.0, 0.5, 0.7, 1.0) # white

        if self.mode == 14:
            self.x = (1.0, 1.0, 1.0, 1.0) # white
            self.turn_right()
            


        if self.x != self.prev_x :
            self.publish_leds(self.x)

        self.prev_x = self.x


        pass

if __name__ == '__main__':
    # create the node
    node = MotionBotNode(node_name='my_publisher_node')

    rate = rospy.Rate(3)
    
    # run node
    while node.gray is None:
        rate.sleep()

    node.publish_leds((1.0, 1.0, 1.0, 1.0))
    
    while not rospy.is_shutdown():
        node.run()
      
    # keep the process from terminating
    rospy.spin()