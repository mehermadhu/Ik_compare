import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_robot_arm(joint_positions,ax):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, len(joint_positions), 10):  
        joint1_position, joint2_position, joint3_position, joint4_position, joint5_position, end_effector_position = joint_positions[i]
        ax.plot([0, joint1_position[0]],[ 0, joint1_position[1]], [0, joint1_position[2]], color='r')
        ax.plot([joint1_position[0], joint2_position[0]], [joint1_position[1], joint2_position[1]], [joint1_position[2], joint2_position[2]], color='k')
        ax.plot([joint2_position[0], joint3_position[0]], [joint2_position[1], joint3_position[1]], [joint2_position[2], joint3_position[2]], color='b')
        ax.plot([joint3_position[0], joint4_position[0]], [joint3_position[1], joint4_position[1]], [joint3_position[2], joint4_position[2]], color='g')
        ax.plot([joint4_position[0], joint5_position[0]], [joint4_position[1], joint5_position[1]], [joint4_position[2], joint5_position[2]], color='c')
        ax.plot([joint5_position[0], end_effector_position[0]], [joint5_position[1], end_effector_position[1]], [joint5_position[2], end_effector_position[2]], color='m')
        #print(f'x:{end_effector_position[0]},y:{end_effector_position[1]},z:{end_effector_position[2]}')
    x_values = []
    y_values = []
    z_values = []

    for joint_position in joint_positions:
        for point in joint_position:
            x_values.append(point[0])
            y_values.append(point[1])
            z_values.append(point[2])
    min_lim = min(x_values+y_values+z_values)
    max_lim = max(x_values+y_values+z_values)
        
    ax.set_xlim([min_lim,max_lim])
    ax.set_ylim([min_lim,max_lim])
    ax.set_zlim([min_lim,max_lim])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
   # plt.show()

def plot_data(original_x, original_y, fit_x, fit_y,xlabel,ylabel):
    plt.figure()
    plt.scatter(original_x, original_y, label='original data')
    # plot the fit
    plt.plot(fit_x, fit_y, color='red', label='fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()