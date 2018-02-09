from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, *args, **kwargs):
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']

        fuel_capacity = kwargs['fuel_capacity']
        steer_ratio = kwargs['steer_ratio']
        vehicle_mass = kwargs['vehicle_mass']
        wheel_radius = kwargs['wheel_radius']
        wheel_base = kwargs['wheel_base']
        max_lat_accel = kwargs['max_lat_accel']
        max_steer_angle = kwargs['max_steer_angle']
        min_speed = 0.001

        params = [wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle]
        self.yaw_controller = YawController(*params)
        self.lowpass = LowPassFilter(3.0, 1.0)
        self.brake_torque = (vehicle_mass + fuel_capacity * GAS_DENSITY) * wheel_radius
        self.velocity_pid = PID(kp=1.4, ki=0, kd=0, mn=self.decel_limit, mx=self.accel_limit)
        self.steer_pid = PID(kp=0.7, ki=0.004, kd=0.3, mn=-max_steer_angle, mx=max_steer_angle)
        self.last_time = rospy.get_time()

    def control(self, *args, **kwargs):
        linear_setpoint = kwargs['linear_setpoint']
        angular_setpoint = kwargs['angular_setpoint']
        linear_current = kwargs['linear_current']

        velocity_error = linear_setpoint - linear_current

        now = rospy.get_time()
        delta = now - self.last_time if self.last_time else 0.1
        self.last_time = now
        
        unfiltered = self.velocity_pid.step(velocity_error, delta)
        velocity = self.lowpass.filt(unfiltered)

        rospy.loginfo("linear setpoint = {}, linear current = {}, velocity error = {}, delta = {}, unfiltered pid output = {}, lowpass filtered = {}".format(linear_setpoint, linear_current, velocity_error, delta, unfiltered, velocity))

        steer = self.yaw_controller.get_steering(linear_setpoint, angular_setpoint, linear_current)
        steer = self.steer_pid.step(steer, delta)

        throttle = 0
        brake = 0

        # brake if setpoint velocity less than threshold
        if linear_setpoint < 0.11:
            steer = 0
            brake = abs(self.decel_limit) * self.brake_torque
        else:
            # speed up
            if 0 < velocity:
                throttle = velocity
            # may slow down
            else:
                velocity = abs(velocity)
                # brake if outside deadband
                if self.brake_deadband < velocity:
                    brake = velocity * self.brake_torque

        rospy.loginfo("throttle = {}, brake = {}, steering = {}, linear current = {}".format(throttle, brake, steer, linear_current))

        return throttle, brake, steer
