import numpy as np
import matplotlib.pyplot as plt


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
