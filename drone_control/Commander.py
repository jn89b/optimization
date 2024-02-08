from pymavlink import mavutil
from drone_control.DroneInfo import DroneInfo
from drone_control.DataCollection import DataCollection
import numpy as np
import time 
import math 

def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

class Commander:
    def __init__(self, connection:str='udp:127.0.0.1:14550',
                 frequency:int=10):
        self.connection = connection
        self.vehicle = mavutil.mavlink_connection(connection)
        self.confirm_heartbeat()
        
        self.drone_info = DroneInfo(self.vehicle, frequency)
        
    def confirm_heartbeat(self) -> None:
        print('Waiting for heartbeat')
        self.vehicle.wait_heartbeat()
        print('Heartbeat from system (system %u component %u)' % (self.vehicle.target_system, self.vehicle.target_component))

    
    def get_telem(self):
        return self.drone_info.get_telemetry()
    
    def send_attitude_target(self, roll_angle=0.0, pitch_angle=0.0,
                         yaw_angle=None, yaw_rate=0.0, use_yaw_rate=False,
                         thrust=0.5, body_roll_rate=0.0, body_pitch_rate=0.0):

        master = self.vehicle 

        if yaw_angle is None:
            yaw_angle = master.messages['ATTITUDE'].yaw

        # print("yaw angle is: ", yaw_angle)
        master.mav.set_attitude_target_send(
            0,  # time_boot_ms (not used)
            master.target_system,  # target system
            master.target_component,  # target component
            0b00000000 if use_yaw_rate else 0b00000100,
            to_quaternion(roll_angle, pitch_angle, yaw_angle),  # Quaternion
            body_roll_rate,  # Body roll rate in radian
            body_pitch_rate,  # Body pitch rate in radian
            np.radians(yaw_rate),  # Body yaw rate in radian/second
            thrust
        )

    
