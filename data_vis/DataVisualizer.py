import numpy as np
import matplotlib.pyplot as plt

#animation imports
from matplotlib import animation
from mpl_toolkits import mplot3d

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
    def plot_controls(solution:dict, time_list:np.ndarray, n_controls:int,
                      add_one_more:bool=False, additional_row:list=None,
                      label_name:str=None) -> tuple:
        #plot controls
        if add_one_more == True:
            fig,ax = plt.subplots(nrows=n_controls+1, figsize=(10,10))
        else:
            fig,ax = plt.subplots(nrows=n_controls, figsize=(10,10))
        u_phi = np.rad2deg(solution['u_phi'])
        u_theta = np.rad2deg(solution['u_theta'])
        u_psi = np.rad2deg(solution['u_psi'])
        v_cmd = solution['v_cmd']
        
        #check if time_list is one less than the control list
        if len(time_list) == len(u_phi):
            time_list = time_list
        else:
            time_list = time_list[:-1]
                
    
        ax[0].plot(time_list, u_phi, 'r', label='u_phi')
        ax[1].plot(time_list, u_theta, 'g', label='u_theta')
        ax[2].plot(time_list, u_psi, 'b', label='u_psi'), 
        ax[3].plot(time_list, v_cmd, 'k', label='v_cmd')
        ax[4].plot(time_list, additional_row, 'm', label=label_name)

        print("minimal distance: ", min(additional_row))
        
        for ax in ax:
            ax.set_ylabel('Control')
            ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid()
        
        # if add_one_more:
        #     ax[4].set_ylabel(label_name)
        
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
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
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
    
    
    def animate_trajectory_3D(self, solution:dict, use_time_span:bool=True, time_span:int=5,
                              animation_interval:int=20,
                            time_list:np.ndarray=None, 
                            show_velocity:bool=False,
                            vel_min:float=0, vel_max:float=1) -> tuple:
        """
        Animate the 3D trajectory
        """
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10,10))
        x = solution['x']
        y = solution['y']
        z = solution['z']
        velocity = solution['v_cmd']
        #animate lines 
        lines = []
        lines = [ax.plot([], [], [], 'r', label='3D Trajectory')[0]]
        
        pts = []
        pts =[ax.plot([], [], [], c='b', marker='o', label='Ego Position')]
        
    
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            for pt in pts:
                pass
                # pt.set_data([], [])
                # pt.set_3d_properties([])
            return lines
        
        if show_velocity:
            colorMap = plt.cm.get_cmap('jet')
            #normalize the velocity
            norm = plt.Normalize(vel_min, vel_max)
            colors = colorMap(norm(velocity))
            
            
        def update(frame):
            for line,scatter in zip(lines,pts):
                
                if frame < time_span:
                    idx_interval = 0
                else:
                    idx_interval = frame - time_span
                
                if show_velocity:
                    #set the color of the line based on the velocity
                    line.set_color(colors[frame])
                    #set scatter color
                    # scatter.set_color(colors[frame])
                
                if use_time_span:
                    line.set_data(x[idx_interval:frame], y[idx_interval:frame])
                    line.set_3d_properties(z[idx_interval:frame])
                    # scatter.set_data(x[frame], y[frame])
                    # scatter.set_3d_properties(z[frame])
                                            
                else:
                    line.set_data(x[:frame], y[:frame])
                    line.set_3d_properties(z[:frame])
                    # scatter.set_data(x[frame], y[frame])
                    # scatter.set_3d_properties(z[frame])
                                
            return lines
        
        ani = animation.FuncAnimation(fig, update, frames=len(x), interval=animation_interval,
                                      init_func=init, blit=True)

        #show color bar for velocity
        if show_velocity:
            sm = plt.cm.ScalarMappable(cmap=colorMap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Velocity m/s')
            
        return fig,ax,ani

    @staticmethod
    def plot_attitudes(solution:dict, time_list:np.ndarray, n_states:int) -> tuple:
        fig,ax = plt.subplots(nrows=n_states, figsize=(10,10))
        
        phi = np.rad2deg(solution['phi'])
        theta = np.rad2deg(solution['theta'])
        psi = np.rad2deg(solution['psi'])
        
        #wrap psi to -180 to 180
        psi = np.rad2deg(np.unwrap(np.deg2rad(psi)))

        ax[0].plot(time_list, np.rad2deg(phi), 'r', label='phi')
        ax[1].plot(time_list, np.rad2deg(theta), 'g', label='theta')
        ax[2].plot(time_list, np.rad2deg(psi), 'b', label='psi')
        ax[3].plot(time_list, solution['v'], 'k', label='v')
                
        for ax in ax:
            ax.set_ylabel('Attitude')
            ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid()
            
        return fig,ax
            
            
    @staticmethod
    def plot_controls_load(solution:dict, time_list:np.ndarray, n_controls:int,
                      add_one_more:bool=False, additional_row:list=None,
                      label_name:str=None) -> tuple:
        #plot controls
        if add_one_more == True:
            fig,ax = plt.subplots(nrows=n_controls+1, figsize=(10,10))
        else:
            fig,ax = plt.subplots(nrows=n_controls, figsize=(10,10))
        load_x = solution['load_x']
        load_z = solution['load_z']
        u_phi = np.rad2deg(solution['u_phi'])
        v_cmd = solution['v_cmd']
        
        #check if time_list is one less than the control list
        if len(time_list) == len(load_x):
            time_list = time_list
        else:
            time_list = time_list[:-1]
                
    
        ax[0].plot(time_list, load_x, 'r', label='load_x')
        ax[1].plot(time_list, load_z, 'g', label='u_theta')
        ax[2].plot(time_list, u_phi, 'b', label='u_psi'), 
        ax[3].plot(time_list, v_cmd, 'k', label='v_cmd')
        ax[4].plot(time_list, additional_row, 'm', label=label_name)

        for ax in ax:
            ax.set_ylabel('Control')
            ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid()
        
        # if add_one_more:
        #     ax[4].set_ylabel(label_name)
        
        return fig,ax
    
    
        
