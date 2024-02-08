from pymavlink import mavutil
import math as m
from drone_control.Telem import Telem
import time 


class DroneInfo():
    def __init__(self, master:mavutil.mavlink_connection, 
                 drone_info_frequency:int=100) -> None:
        
        self.master = master
        self.drone_info_frequency = drone_info_frequency
        self.__start_listening()
        self.start_time = time.time()
        
    def __start_listening(self) -> None:
        self.__request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 
                                    self.drone_info_frequency)
        self.__request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
                                    self.drone_info_frequency)
        self.__request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                                    self.drone_info_frequency)
        self.__request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION,
                                    self.drone_info_frequency)
        
        
    def __get_data(self) -> Telem:
        
        output = Telem()
        msg = self.master.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE', 'GLOBAL_POSITION_INT'], blocking=True)

        try:
            output.lat = self.master.messages['GLOBAL_POSITION_INT'].lat
            output.lon = self.master.messages['GLOBAL_POSITION_INT'].lon
            output.alt = self.master.messages['GLOBAL_POSITION_INT'].alt
            output.heading = self.master.messages['GLOBAL_POSITION_INT'].hdg

            qx = self.master.messages['ATTITUDE_QUATERNION'].q1
            qy = self.master.messages['ATTITUDE_QUATERNION'].q2
            qz = self.master.messages['ATTITUDE_QUATERNION'].q3
            qw = self.master.messages['ATTITUDE_QUATERNION'].q4
            
            output.roll = self.master.messages['ATTITUDE'].roll
            output.pitch = self.master.messages['ATTITUDE'].pitch
            output.yaw = self.master.messages['ATTITUDE'].yaw

            #wrap heading to 0-360
            output.heading = (output.heading + 360) % 360
            output.heading = m.radians(output.heading)
            #wrap yaw from -pi to pi
            if output.heading > m.pi:
                output.heading -= 2*m.pi
            elif output.heading < -m.pi:
                output.heading += 2*m.pi
            # output.yaw = output.heading
            
            output.roll_rate = self.master.messages['ATTITUDE'].rollspeed
            output.pitch_rate = self.master.messages['ATTITUDE'].pitchspeed
            output.yaw_rate = self.master.messages['ATTITUDE'].yawspeed

            output.x = self.master.messages['LOCAL_POSITION_NED'].x
            output.y = self.master.messages['LOCAL_POSITION_NED'].y
            output.z = self.master.messages['LOCAL_POSITION_NED'].z
            output.vx = self.master.messages['LOCAL_POSITION_NED'].vx
            output.vy = self.master.messages['LOCAL_POSITION_NED'].vy
            output.vz = self.master.messages['LOCAL_POSITION_NED'].vz
            output.timestamp = time.time() - self.start_time
            
        #catch key error 
        except KeyError:
            return output

        return output 

    def __request_message_interval(self, message_id:int, frequency_hz:int) -> None:
        """
        Request MAVLink message in a desired frequency,
        documentation for SET_MESSAGE_INTERVAL:
            https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL

        Args:
            message_id (int): MAVLink message ID
            frequency_hz (float): Desired frequency in Hz
        """
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            message_id, # The MAVLink message ID
            1e6 / frequency_hz, # The interval between two messages in microseconds. Set to -1 to disable and 0 to request default rate.
            0, 0, 0, 0, # Unused parameters
            0, # Target address of message stream (if message has target address fields). 0: Flight-stack default (recommended), 1: address of requestor, 2: broadcast.
        )

    def get_telemetry(self) -> Telem:
        return self.__get_data()
    
