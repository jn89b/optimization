import numpy as np
import matplotlib.pyplot as plt

#animation imports
from matplotlib import animation

class DataVisualizer():
    """
    A helper class to visualize the data from the optimization problem
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def plot_states(solution:dict, time_list:np.ndarray, n_states:int) -> tuple:
        #plot states
        fig,ax = plt.subplots(nrows=n_states, figsize=(10,10))
        x = solution['x']
        y = solution['y']
        z = solution['z']
        phi = np.rad2deg(solution['phi'])
        theta = np.rad2deg(solution['theta'])
        psi = np.rad2deg(solution['psi'])
        v = solution['v']
        
        ax[0].plot(time_list, x, 'r', label='x')
        ax[1].plot(time_list, y, 'g', label='y')
        ax[2].plot(time_list, z, 'b', label='z')
        ax[3].plot(time_list, phi, 'k', label='phi')
        ax[4].plot(time_list, theta, 'm', label='theta')
        ax[5].plot(time_list, psi, 'c', label='psi')
        ax[6].plot(time_list, v, 'y', label='v')

        for ax in ax:
            ax.set_ylabel('State')
            ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid()
        
        return fig,ax
    
    @staticmethod
    def plot_controls(solution:dict, time_list:np.ndarray, n_controls:int) -> tuple:
        #plot controls
        fig,ax = plt.subplots(nrows=n_controls, figsize=(10,10))
        u_phi = np.rad2deg(solution['u_phi'])
        u_theta = np.rad2deg(solution['u_theta'])
        u_psi = np.rad2deg(solution['u_psi'])
        v_cmd = solution['v_cmd']
        
        ax[0].plot(time_list[:-1], u_phi, 'r', label='u_phi')
        ax[1].plot(time_list[:-1], u_theta, 'g', label='u_theta')
        ax[2].plot(time_list[:-1], u_psi, 'b', label='u_psi'), 
        ax[3].plot(time_list[:-1], v_cmd, 'k', label='v_cmd')

        for ax in ax:
            ax.set_ylabel('Control')
            ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid()
        
        return fig,ax 

    @staticmethod
    def plot_trajectory_2d(solution:dict, use_time_color:bool=False, 
                        time_list:np.ndarray=None) -> tuple:
        #plot 2d trajectory
        fig, ax = plt.subplots(figsize=(10,10))
        x = solution['x']
        y = solution['y']
        
        if use_time_color:
            ax.scatter(x, y, c=time_list, cmap='viridis', label='2D Trajectory')
            ax.colorbar(label='Time')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid()
        else:
            ax.plot(x, y, 'r', label='2D Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid()
        
        return fig, ax


    @staticmethod
    def plot_trajectory_3d(solution:dict, use_time_color:bool=False,
                        time_list:np.ndarray=None) -> tuple:
        
        #plot 3d trajectory
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10,10))
        x = solution['x']
        y = solution['y']
        z = solution['z']
        
        if use_time_color:
            ax.scatter(x, y, z, c=time_list, cmap='viridis', label='3D Trajectory')
            cbar = plt.colorbar(ax.scatter(x, y, z,  c=time_list, 
                                        cmap='viridis', marker='x'))
            

            #ax.colorbar(label='Time')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.grid()
        else:
            ax.plot(x, y, z, 'r', label='3D Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.grid()
            
        return fig, ax


    def plot_obstacles_2D(obs_params:dict, current_ax) -> tuple:

        x_list = obs_params['x']
        y_list = obs_params['y']
        radii_list = obs_params['radii']
        
        for x,y,r in zip(x_list, y_list, radii_list):
            circle = plt.Circle((x, y), r, color='b', fill=False)
            current_ax.add_artist(circle)
            
        return current_ax

    @staticmethod
    def plot_obstacles_3D(obs_params:dict, current_ax,
                        z_low:float=0, z_high:float=10,
                        num_points:int=20,
                        color_obs:str='b',
                        alpha_val:float=0.4):
        """
        Plot as a cylinder
        """
        
        x_list = obs_params['x']
        y_list = obs_params['y']
        radii_list = obs_params['radii']
        
        for x,y,r in zip(x_list, y_list, radii_list):
            # Cylinder parameters
            radius = r
            #height = z_high - z_low
            center = [x, y]
            
            theta = np.linspace(0, 2*np.pi, num_points)
            z = np.linspace(z_low, z_high, num_points)
            
            theta, z = np.meshgrid(theta, z)
            x_vector = radius * np.cos(theta) + center[0]
            y_vector = radius * np.sin(theta) + center[1]
            
            current_ax.plot_surface(x_vector, y_vector, z, 
                            color=color_obs, alpha=alpha_val)
            
        return current_ax


    @staticmethod
    def unpack_solution_list(solution_list:dict) -> dict:
        """
        Unpack the solution list into a dictionary of arrays
        """
        solution = {}
        for key in solution_list[0].keys():
            solution[key] = np.array([sol[key] for sol in solution_list])
                
        #flatten the arrays
        for key in solution.keys():
            if key != 'time':
                solution[key] = solution[key].flatten()
            else:
                solution[key] = solution[key].flatten()
        
        return solution
    
    
    def animate_trajectory_3D(self, solution:dict, time_span:int=5,
                              animation_interval:int=20,
                            time_list:np.ndarray=None):
        """
        Animate the 3D trajectory
        """
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10,10))
        x = solution['x']
        y = solution['y']
        z = solution['z']
        
        #animate lines 
        lines = []
        #check if x is a single list
        # if isinstance(x, list):
        #     x = np.array(x)
        #     y = np.array(y)
        #     z = np.array(z)
        lines = [ax.plot([], [], [], 'r', label='3D Trajectory')[0]]
        
        # for i in range(len(x)-1):
        #     line, = ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 'r', label='3D Trajectory')
        #     lines.append(line)
            
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines
        
        def update(frame):
            for line in lines:
                
                if frame < time_span:
                    idx_interval = 0
                else:
                    idx_interval = frame - time_span
                
                line.set_data(x[idx_interval:frame], y[idx_interval:frame])
                line.set_3d_properties(z[idx_interval:frame])
            return lines
        
        ani = animation.FuncAnimation(fig, update, frames=len(x), interval=animation_interval,
                                      init_func=init, blit=True)


        return fig,ax,ani

    
        
