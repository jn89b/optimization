
# First Task Install stuff for trajectory
## Install ThirdParty-HSL
```
https://github.com/coin-or-tools/ThirdParty-HSL.git
```

- Follow the ##Installation steps in this directory and put the coinhsl file into the inner folder of the third_party HSL 
- https://github.com/coin-or-tools/ThirdParty-HSL
- You're going to want to extract the file that I sent you and dump it into the respective location


## Install Casadi from source 
Follow the readme.md from this document to install casadi from source 
https://github.com/jn89b/optimization/blob/master/readme.md


## Set up the python virtual environment
https://github.com/jn89b/optimization/blob/master/readme.md

## Run the script
- Run the sim.py script and let me know if you get any plots, if you have any issues let me know 


## Second Task Control quadcopter/fixed wing with pymavutil
- On your main computer I would like you to start a simulation with Ardupilot SITL using a fixed wing system,

- From there takeoff your fixed wing aircraft in the simulation utilizing whatever commands and set the fixed wing aircraft to Guided Mode

- Once you are good create a python script from the Jetson to have the fixed wing fly, you can refer to the sine_wave_test.py script to get an idea of how this works. 

- In addition you might need to install pymavutil
- To figure out how this all works check out https://mavlink.io/en/mavgen_python/ for scripting the drone to fly autonomously
- Let me know if you have any questions and thanks again for your help! 

```python
    
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

```