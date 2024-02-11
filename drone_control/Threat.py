import numpy as np 

"""
Threat class for the drone

Initialize parameters for 

"""

def wrap_radian_angle(angle:float) -> float:
    """
    Wrap the angle to be between -pi and pi
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


#move this file to somewhere else that makes more since
class Threat():
    def __init__(self, init_position:np.ndarray, algos_params:dict, 
                 threat_params:dict, 
                 use_2D:bool=True,
                 use_3D:bool=False) -> None:
        """
        #2D consist of x, y, and psi
        #3D consist of x, y, z, and psi
        """
        self.use_2D = use_2D
        self.use_3D = use_3D
        self.init_position = init_position
        self.threat_params = threat_params
        self.threat_velocity = threat_params['velocity'] #m/s
        self.threat_type = threat_params['type'] #holonomic or non-holonomic
        
        
        #check to make sure we both don't use 2D and 3D
        if self.use_2D and self.use_3D:
            raise ValueError('Cannot use both 2D and 3D, Figure out your life')
        
        #check to make sure we use either 2D or 3D
        if not self.use_2D and not self.use_3D:
            raise ValueError('Must use either 2D or 3D, Figure out your life')
        
        self.time_traj = None
        if self.use_2D:
            if self.init_position.size != 3:
                raise ValueError('Initial position must be of size 3')
            self.init_position = init_position
            self.x_traj = None
            self.y_traj = None
            self.psi_traj = None
            
            # self.straight_line_2D(algos_params['final_position'], algos_params['num_points'])
            
        if self.use_3D:
            if self.init_position.size != 4:
                raise ValueError('Initial position must be of size 4')
            self.init_position = init_position
            self.x_traj = None
            self.y_traj = None
            self.z_traj = None
            self.psi_traj = None
        
    def simulate_dynamics(self, current_position:np.ndarray, controls:np.ndarray, dt:float) -> np.ndarray:
        """
        Simulate the dynamics of the threat
        """
        #simple euler's method
        velocity = controls[0]
        ang_vel = controls[1]
        
        if self.threat_type == 'holonomic' and self.use_2D:
            #simulate the threat as a holonomic threat
            x = current_position[0] + velocity * np.cos(self.init_position[2]) * dt
            y = current_position[1] + velocity * np.sin(self.init_position[2]) * dt
            psi = self.init_position[2] + (ang_vel * dt)
            return np.array([x, y, psi])
            
        if self.threat_type == 'non-holonomic' and self.use_2D:
            x = current_position[0] + velocity * np.cos(current_position[2]) * dt
            y = current_position[1] + velocity * np.sin(current_position[2]) * dt
            psi = current_position[2] + (ang_vel * dt)
            return np.array([x, y, psi])
        
        if self.threat_type == 'holonomic' and self.use_3D:
            #simulate the threat as a holonomic threat
            x = current_position[0] + velocity * np.cos(self.init_position[3]) * dt
            y = current_position[1] + velocity * np.sin(self.init_position[3]) * dt
            z = current_position[2] + velocity * np.sin(self.init_position[4]) * dt
            psi = self.init_position[3] + (ang_vel * dt)
            return np.array([x, y, z, psi])
            
        if self.threat_type == 'non-holonomic' and self.use_3D:
            x = current_position[0] + velocity * np.cos(current_position[3]) * dt
            y = current_position[1] + velocity * np.sin(current_position[3]) * dt
            z = current_position[2] + velocity * np.sin(current_position[4]) * dt
            psi = current_position[3] + (ang_vel * dt)
            return np.array([x, y, z, psi])    
        
    def straight_line_2D(self, final_position:np.ndarray, num_points:int, dt:float) -> np.ndarray:
        """
        Initialize the 2D trajectory for the threat
        """
        if final_position.size != 3:
            raise ValueError('Final position must be of size 3')
        
        x_traj = []
        y_traj = []
        psi_traj = []
        
        
        start_position = self.init_position
        if self.threat_type == 'holonomic':
            controls = [self.threat_velocity, 0]
            #simulate the threat as a holonomic threat
            for i in range(num_points+1):
                position = self.simulate_dynamics(
                    start_position, controls, dt)
                # position[2] = wrap_radian_angle(position[2])

                x_traj.append(position[0])
                y_traj.append(position[1])
                psi_traj.append(position[2])
                start_position = position        
        else:            
            for i in range(num_points+1):
                position = self.simulate_dynamics(
                    start_position, controls, dt)
        
                position[2] = wrap_radian_angle(position[2])
                x_traj.append(position[0])
                y_traj.append(position[1])
                psi_traj.append(position[2])
                start_position = position
                
        self.x_traj = x_traj
        self.y_traj = y_traj
        self.psi_traj = psi_traj
        self.time_traj = np.arange(0, num_points*dt, dt)
        
    def straight_line_3D(self, final_position:np.ndarray, num_points:int, dt:float) -> np.ndarray:
        """
        Initialize the 3D trajectory for the threat
        """
        if final_position.size != 4:
            raise ValueError('Final position must be of size 4')
        if self.threat_type == 'holonomic':
            #simulate the threat as a holonomic threat
            x = self.threat_velocity * np.cos(self.init_position[3]) * np.linspace(0, num_points, num_points)
            y = self.threat_velocity * np.sin(self.init_position[3]) * np.linspace(0, num_points, num_points)
            z = self.threat_velocity * np.sin(self.init_position[4]) * np.linspace(0, num_points, num_points)
            psi = np.linspace(self.init_position[3], self.init_position[3], num_points)
        else:
            x = np.linspace(self.init_position[0], final_position[0], num_points)
            y = self.threat_velocity * np.sin(self.init_position[3]) * np.linspace(0, num_points, num_points)
            z = self.threat_velocity * np.sin(self.init_position[4]) * np.linspace(0, num_points, num_points)
            psi = np.linspace(self.init_position[3], self.init_position[3], num_points)
        
        # return np.vstack((x, y, z, psi))
        self.x_traj = x
        self.y_traj = y
        self.z_traj = z
        self.psi_traj = psi
        
        
    
    
    

    