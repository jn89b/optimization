import numpy as np
import matplotlib.pyplot as plt
import time 
from drone_control.Commander import Commander
from drone_control.DataCollection import DataCollection

def generate_sine(amplitude, frequency, time_interval, N):
    """
    Plots a sine wave based on the specified amplitude, frequency, and time interval.

    Parameters:
    amplitude (float): The peak value (maximum) of the wave.
    frequency (float): The number of cycles the wave completes in one second.
    time_interval (float): The total time period over which to plot the wave, in seconds.
    N (int): The number of points to plot within the time interval.
    """
    # Time array
    t = np.linspace(0, time_interval, N)  # 1000 points within the time interval
    # Sine wave formula
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return t, y    


# # Example usage
commander = Commander(frequency=10)
folder_dir = 'flight_data/train/'
time.sleep(5)
number_of_intervals = 3
random_pitch = np.random.uniform(5, 15)
# pitch_cmds = [15, 10, 5]
for i in range(number_of_intervals):
    file_name = folder_dir + f'train_{i}.csv'
    data_collection = DataCollection(file_name)
    telem = commander.get_telem()
    random_pitch += -1
    random_frequency = np.random.uniform(0.5, 1.5)
    t, pitch_cmds = generate_sine(amplitude=random_pitch, frequency=random_frequency, time_interval=5.0, N=300)
    data_collection.add_data(telem)
    print("collecting data... for interval: ", i)
    for j, cmd in enumerate(pitch_cmds):
        #random roll
        random_roll = np.random.uniform(-40, 40)
        
        #this takes in degrees as commands
        commander.send_attitude_target(roll_angle=0.0, pitch_angle=cmd, 
                                    yaw_angle=0.0)
        
        telem = commander.get_telem()
        data_collection.add_data(telem)
        print('Pitch', np.rad2deg(telem.pitch))
        print('Roll', np.rad2deg(telem.roll))
        print(j)
        print("\n")

    data_collection.to_csv()
        