#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import PIL.Image as PILImage
import tf as ros_tf
import cv2
import yaml
import numpy as np
import math
import sys

MAX_DISTANCE = 100.0
MIN_DISTANCE = 0.0
MAX_ANGLE = 15.0 * (math.pi/180.0)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.base_waypoints = None        
        self.current_pos = None
        self.current_yaw = None

        self.stop_lines = None    # waypoint coordinates that correspond to each traffic light [[x1, y1]..[xn, yn]]
        self.stop_waypoint = None
        self.closest_next_tl = -1 # id of the closest traffic light
        self.classifier = TLClassifier()

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)
        self.tl_publisher = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.stop_lines = np.array(config["stop_line_positions"])

        rospy.spin()

    def pose_cb(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.current_pos = (position.x, position.y)
        self.current_yaw = ros_tf.transformations.euler_from_quaternion(quaternion)[2]
        if self.base_waypoints is not None:
            self.closest_next_tl = self.get_next_closest_tl()

    def waypoints_cb(self, msg):
        base_waypoints = [np.array([waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]) for waypoint in msg.waypoints]
        self.base_waypoints = np.array(base_waypoints)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        if self.get_next_closest_tl() == -1:
            rospy.logdebug("no traffic lights. no images processed.")
            self.tl_publisher.publish(Int32(-1))
            return

        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")[...,::-1]
        image = PILImage.fromarray(np.uint8(cv_image))
        image_np = np.expand_dims(image, axis=0)
        result = self.classifier.get_classification(image_np)

        if (result == TrafficLight.RED) or (result == TrafficLight.YELLOW):
            self.save_stop_waypoint_index()
            self.tl_publisher.publish(Int32(self.stop_waypoint))
        elif result == TrafficLight.GREEN:
            self.tl_publisher.publish(Int32(-3))
        elif result == TrafficLight.UNKNOWN:
            self.tl_publisher.publish(Int32(-2))
        else:
            self.tl_publisher.publish(Int32(-1))

    def save_stop_waypoint_index(self):
        if 0 <= self.closest_next_tl:
            tl_id = self.closest_next_tl
            min_distance = sys.maxint
            for i, waypoint in enumerate(self.base_waypoints):
                distance = self.euclid_distance(self.stop_lines[tl_id][0], waypoint[0], 
                                                self.stop_lines[tl_id][1], waypoint[1])
                if distance < min_distance:
                    min_distance = distance
                    self.stop_waypoint = i
        else:
            self.stop_waypoint = -1

    def get_next_closest_tl(self):
        """
        return the id of the next traffic light or -1 if it is too far.
        """

        if (self.stop_lines is not None) and (self.current_pos is not None):
            for i, tl in enumerate(self.stop_lines):
                distance = self.euclid_distance(tl[0], self.current_pos[0], tl[1], self.current_pos[1])
                direction = math.atan2(tl[1] - self.current_pos[1] , tl[0] - self.current_pos[0])
                angle_diff = math.atan2(math.sin(direction - self.current_yaw), math.cos(direction - self.current_yaw))

                if (MIN_DISTANCE < distance) and (distance < MAX_DISTANCE) and (abs(angle_diff) < MAX_ANGLE):
                    return i
        return -1

    def euclid_distance(self, x1, x2, y1, y2):
        a = np.array((x1, y1))
        b = np.array((x2, y2))
        return np.linalg.norm(a - b)

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
