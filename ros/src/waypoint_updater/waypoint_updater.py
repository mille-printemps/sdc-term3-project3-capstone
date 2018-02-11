#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np
import tf as ros_tf
import math
import copy
import sys
from collections import namedtuple

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

RATE = 50 # in Hz
LOOKAHEAD_WAYPOINTS = 100 # number of waypoints to be published
KPH_TO_MPS = 1.0/3.6

MAX_VELOCITY = KPH_TO_MPS * rospy.get_param("/waypoint_loader/velocity")
GO = 0.8
SLOW = 0.3

LIMIT_DISTANCE = 100
BRAKE_DISTANCE = 55
HARD_LIMIT_DISTANCE = 8

#DEBUG = True
DEBUG = False

Behavior = namedtuple("Behavior", ["KEEP", "GO", "STOP", "SLOW", "BRAKE"])
BEHAVIOR = Behavior(KEEP = 0, GO = 1, STOP = 2, SLOW = 3, BRAKE = 4)

LOG_LEVEL = rospy.INFO

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=LOG_LEVEL)

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Waypoint, obstacle_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.current_pose = None
        self.current_yaw = None
        self.current_velocity = None

        self.waypoint_saved_velocity = None
        self.tl_waypoint_id = None

        self.update()
        rospy.spin()

    # find index of nearest waypoint and publish the next waypoints
    def update(self):
        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown():
            if self.base_waypoints and self.current_pose and self.current_velocity:
                self.find_nearest_waypoint()
                self.publish_next_waypoints()
            rate.sleep()

    def find_nearest_waypoint(self):
        nearest_waypoint = [-1, sys.maxint]
        car_coord = self.current_pose.pose.position

        for i in range(len(self.base_waypoints)):
            wp_coord = self.base_waypoints[i].pose.pose.position
            distance = self.euclid_distance(car_coord.x, wp_coord.x, car_coord.y, wp_coord.y)
            direction = math.atan2(car_coord.y - wp_coord.y, car_coord.x - wp_coord.x)
            angle_diff = math.atan2(math.sin(direction - self.current_yaw), math.cos(direction - self.current_yaw))
            if (distance < nearest_waypoint[1]) and (abs(angle_diff) < math.pi / 4.0):
                nearest_waypoint = [i, distance]
        self.current_waypoint_id = nearest_waypoint[0]
        rospy.logdebug("current waypoint id = {}".format(self.current_waypoint_id))

    def publish_next_waypoints(self):
        waypoints = Lane()
        waypoints.header.stamp = rospy.Time(0)
        waypoints.header.frame_id = self.current_pose.header.frame_id

        if (self.current_waypoint_id + LOOKAHEAD_WAYPOINTS) < len(self.base_waypoints):
            waypoints.waypoints = copy.deepcopy(self.base_waypoints[self.current_waypoint_id: 
                                                                    self.current_waypoint_id + LOOKAHEAD_WAYPOINTS])
        else:
            front = copy.deepcopy(self.base_waypoints[self.current_waypoint_id:])
            back = copy.deepcopy(self.base_waypoints[:LOOKAHEAD_WAYPOINTS - len(front)])
            waypoints.waypoints = front + back

        waypoints.waypoints = self.behavior(waypoints.waypoints)
        self.final_waypoints_pub.publish(waypoints)
        rospy.logdebug("waypoints updated")

    def pose_cb(self, msg):
        self.current_pose = msg
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.current_yaw = ros_tf.transformations.euler_from_quaternion(quaternion)[2]

    def velocity_cb(self, msg):
        self.current_velocity = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints.waypoints
        self.base_waypoints_sub.unregister()

    def obstacle_cb(self, msg):
        # TODO
        pass

    def traffic_cb(self, msg):
        self.tl_waypoint_id = msg.data

    def reset_velocity(self, waypoints, velocity):
        self.waypoint_saved_velocity = None
        for i in range(LOOKAHEAD_WAYPOINTS):
            waypoints[i].twist.twist.linear.x = velocity
        return waypoints

    def update_velocity(self, waypoints, multiplier = 1.0):
        self.waypoint_saved_velocity = None
        for i in range(LOOKAHEAD_WAYPOINTS):
            waypoints[i].twist.twist.linear.x *= multiplier
        return waypoints

    def brake(self, waypoints):
        stop_waypoint = self.tl_waypoint_id
        current_velocity = self.current_velocity.twist.linear.x

        rospy.loginfo("breaking at stop waypoint index = {} and current velocity = {}".format(stop_waypoint, current_velocity))

        distance_to_stop_waypoint = self.waypoint_distance(self.current_waypoint_id, stop_waypoint)

        rospy.loginfo("distance to stop_waypoint = {}".format(distance_to_stop_waypoint))

        waypoint_to_stop = min(stop_waypoint - self.current_waypoint_id, LOOKAHEAD_WAYPOINTS)

        rospy.loginfo("waypoints until stop = {} ".format(waypoint_to_stop))

        if self.waypoint_saved_velocity is None:
            self.waypoint_saved_velocity = current_velocity

        for i in range(waypoint_to_stop, LOOKAHEAD_WAYPOINTS):
            waypoints[i].twist.twist.linear.x = 0.0

        for i in range(waypoint_to_stop):
            waypoint_distance = self.waypoint_distance(self.current_waypoint_id + i, stop_waypoint)
            waypoint_velocity = self.waypoint_saved_velocity * math.sqrt(waypoint_distance / distance_to_stop_waypoint)
            waypoints[i].twist.twist.linear.x = waypoint_velocity

        self.waypoint_saved_velocity = waypoints[1].twist.twist.linear.x
        return waypoints

    def decide_behavior(self):
        # red or yellow light
        if 0 <= self.tl_waypoint_id:
            distance_to_tl = self.waypoint_distance(self.current_waypoint_id, self.tl_waypoint_id)

            rospy.loginfo("distance to traffic light = {} ".format(distance_to_tl))

            if LIMIT_DISTANCE < distance_to_tl:
                return BEHAVIOR.KEEP
            elif BRAKE_DISTANCE < distance_to_tl < LIMIT_DISTANCE:
                return BEHAVIOR.GO
            elif HARD_LIMIT_DISTANCE < distance_to_tl < BRAKE_DISTANCE:
                return BEHAVIOR.BRAKE
            else:
                return BEHAVIOR.STOP
        # no traffic lights
        elif self.tl_waypoint_id == -1:
            return BEHAVIOR.KEEP
        # traffic lights unknown
        elif self.tl_waypoint_id == -2:
            return BEHAVIOR.SLOW
        # green light
        elif self.tl_waypoint_id == -3:
            return BEHAVIOR.GO
        else:
            rospy.logerr("UNKOWN BEHAVIOR, this might be a bug")

    def behavior(self, waypoints):
        if DEBUG:
            return waypoints

        # wait for the first traffic light message
        if self.tl_waypoint_id is None:
            rospy.logdebug("waiting for the first traffic light message...")
            return self.reset_velocity(waypoints, 0.0)
        
        # traffic light message received
        behavior = self.decide_behavior()

        if behavior == BEHAVIOR.KEEP:
            rospy.loginfo("no traffic light detected")
            return self.update_velocity(waypoints)
        elif behavior == BEHAVIOR.GO:
            rospy.loginfo("green light detected")
            return self.update_velocity(waypoints, GO)
        elif behavior == BEHAVIOR.STOP:
            rospy.loginfo("red light detected closely")
            return self.reset_velocity(waypoints, 0.0)
        elif behavior == BEHAVIOR.SLOW:
            rospy.loginfo("unknown traffic light detected")
            return self.update_velocity(waypoints, SLOW)
        else:
            rospy.loginfo("red light detected")
            return self.brake(waypoints)

    def waypoint_distance(self, wp1, wp2):
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        distance = 0
        start, end, sign = (wp1, wp2, 1) if wp1 <= wp2 else (wp2, wp1, -1)
        for i in range(end - start):
            distance += dl(self.base_waypoints[start + i].pose.pose.position,
                           self.base_waypoints[start + i + 1].pose.pose.position)
        return sign * distance

    def euclid_distance(self, x1, x2, y1, y2):
        a = np.array((x1, y1))
        b = np.array((x2, y2))
        return np.linalg.norm(a - b)

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
