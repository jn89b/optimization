from pymavlink import mavutil

# Create a connection to the autopilot
# Adjust this connection string to your setup, for example:
# - For a serial connection: '/dev/ttyACM0'
# - For a UDP connection to a companion computer: 'udpin:0.0.0.0:14550'
# - For a TCP connection (less common): 'tcp:192.168.1.2:5760'
connection_string = '/dev/ttyACM0'
master = mavutil.mavlink_connection(connection_string)
master.wait_heartbeat()
pew_pew_param_name ="PEW_PEW"

#do a tolerance check
MIDDLE_VALUE = 1500
ON = 1
OFF = 0

def get_rc_channel_7_value(param_name:str='PEW_PEW'):
    """
    Fetches the current value of RC channel 7.
    
    :return: The current value of channel 7 or None if not available.
    """
    # Wait for the first heartbeat to ensure the connection is established

    while True:
        try:
            # Fetch messages of type 'RC_CHANNELS'
            message = master.recv_match(type='RC_CHANNELS', blocking=True, timeout=1)
            if message:
                # RC channel values are indexed from 1, but attributes in the message are 0-indexed
                channel_7_value = message.chan7_raw  # channel 7 value is in chan6_raw (0-indexed)
                pew_pew_param_value = get_parameter(param_name)
                # print(f"Channel 7 Value: {channel_7_value}")
                # print(f"PEW PEW Value: {pew_pew_param_value}")
                if channel_7_value < MIDDLE_VALUE:
                    print("Channel 7 is less than middle value", channel_7_value)
                    set_parameter(param_name, OFF)
                else:
                    print("Channel 7 is greater than middle value", channel_7_value)
                    set_parameter(param_name, ON)
                    
                # return channel_7_value
        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


# Function to request and fetch a parameter
def get_parameter(param_name):
    """
    Request and return the value of a specific parameter.
    :param param_name: The name of the parameter to fetch.
    :return: The value of the requested parameter, or None if not found.
    """
    # Request the parameter
    master.mav.param_request_read_send(
        master.target_system, master.target_component,
        param_name.encode(), -1
    )

    # Wait for the response from the autopilot
    while True:
        try:
            message = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)
            # print(f"Message: {message}")
            if message is not None:
                # print("parameter is ", message.param_value)
                return message.param_value
            # if message is not None and message.param_id.decode() == param_name:
            #     print(f"Message: {message}" )
                # print(f"Parameter: {param_name}, Value: {message.param_value}")
                # return message.param_value
        except Exception as e:
            print(f"Error retrieving parameter: {e}")
            return None

def set_parameter(param_name:str='PEW_PEW', param_value:int=0) -> None:
    """
    Sets the value of a specified parameter.
    :param param_name: The name of the parameter to set.
    :param param_value: The new value for the parameter.
    """
    print(f"Setting {param_name} to {param_value}")
    # Send the PARAM_SET command with the specified value
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        param_name.encode(),
        param_value,
        mavutil.mavlink.MAV_PARAM_TYPE_INT8
    )

    # Wait for confirmation from the vehicle
    # This loop waits for the PARAM_VALUE message that confirms the change
    while True:
        message = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=10)

        if message is not None and message.param_id == param_name:
            # print(f"Confirmed {param_name} = {message.param_value}")
            break

# Example usage
if __name__ == '__main__':
    get_rc_channel_7_value()
    