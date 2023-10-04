import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

debug  = False

def dh_transform(a, d, alpha, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]])

  
def calculate_joint_positions(joint_angles, a, d, alpha):
    joint_positions = []
    for joint_angle in joint_angles:
        # Calculate each Joint Transformation matrix
        T1 = dh_transform(a[0], d[0], alpha[0], joint_angle[0])  
        T2 = dh_transform(a[1], d[1], alpha[1], joint_angle[1])
        T3 = dh_transform(a[2], d[2], alpha[2], joint_angle[2])

        joint1_position = np.array([0, 0, 0])
        joint2_position = np.matmul(T1, np.array([0, 0, 0, 1]))[:-1]
        joint3_position = np.matmul(np.dot(T1, T2), np.array([0, 0, 0, 1]))[:-1]
        wrist_position = np.matmul(np.dot(np.dot(T1, T2), T3), np.array([0, 0, 0, 1]))[:-1]

        joint_positions.append((joint1_position, joint2_position, joint3_position, wrist_position))
    return joint_positions
    
def plot_robot_arm(joint_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for i in range(0, len(joint_positions), 10):  # take every 10th position for visualization
        joint1_position, joint2_position, joint3_position, wrist_position = joint_positions[i]

        ax.plot([joint1_position[0], joint2_position[0]], [joint1_position[1], joint2_position[1]], [joint1_position[2], joint2_position[2]], color='k')
        ax.plot([joint2_position[0], joint3_position[0]], [joint2_position[1], joint3_position[1]], [joint2_position[2], joint3_position[2]], color='b')
        ax.plot([joint3_position[0], wrist_position[0]], [joint3_position[1], wrist_position[1]], [joint3_position[2], wrist_position[2]], color='r')
    plt.show()
    

def move_around(a,d,alpha):
    # to store wrist center positions
    wrist_center_positions = []
    # to store joint angles
    joint_angles = []

    lower_limit = 0
    upper_limit = np.pi

    # fixed values for theta2 and theta3
    theta2 = 0 
    theta3 = 0

    # increasing the base joint angle
    for theta in np.linspace(lower_limit, upper_limit, num=100):
        theta1 = theta

        T1 = dh_transform(a[0], d[0], alpha[0], theta1)
        T2 = dh_transform(a[1], d[1], alpha[1], theta2)
        T3 = dh_transform(a[2], d[2], alpha[2], theta3)

        # Calculate the transformation to the wrist center.
        T = np.linalg.multi_dot([T1, T2, T3])

        wrist_center_positions.append(T[:-1, -1])  # store only the translation part.
        joint_angles.append((theta1, theta2, theta3))

    # converting list to numpy array for convenience
    wrist_center_positions = np.array(wrist_center_positions)
    joint_angles = np.array(joint_angles)

    # Now `wrist_center_positions` stores the wrist center's positions and
    # `joint_angles` stores the corresponding joint angles.

    return joint_angles, wrist_center_positions

def move_up(a,d,alpha):
    # to store wrist center positions
    wrist_center_positions = []
    # to store joint angles
    joint_angles = []

    lower_limit = 0
    upper_limit = np.pi/2

    # fixed values for theta2 and theta3
    
    theta1 = 0 
    theta3 = 0
    # find order of rotation
    joint2_order = 1 if a[1] * a[2] > 0 else -1
    # increasing the base joint angle
    for theta in np.linspace(lower_limit, upper_limit, num=100):
        theta2 = theta
        T1 = dh_transform(a[0], d[0], alpha[0], theta1)
        T2 = dh_transform(a[1], d[1], alpha[1], theta2*joint2_order)
        T3 = dh_transform(a[2], d[2], alpha[2], theta3)

        # Calculate the transformation to the wrist center.
        T = np.linalg.multi_dot([T1, T2, T3])

        wrist_center_positions.append(T[:-1, -1])  # store only the translation part.
        joint_angles.append((theta1, theta2, theta3))

    # converting list to numpy array for convenience
    wrist_center_positions = np.array(wrist_center_positions)
    joint_angles = np.array(joint_angles)

    # Now `wrist_center_positions` stores the wrist center's positions and
    # `joint_angles` stores the corresponding joint angles.

    return joint_angles, wrist_center_positions
     
def stretch_along(a,d,alpha):
    # Joint limits
    joint2_limits = [0, np.pi/2]
    joint3_limits = [0, np.pi/2]
    joint_angles = []
    wrist_positions = []

    joint1_angle = 0
    joint2_order = -1 if a[1] * a[2] > 0 else 1
    joint3_order = -1 if alpha[1] * alpha[2] < 0 else 1

    T1 = dh_transform(a[0], d[0], alpha[0], joint1_angle)
    
    # sweep joint2 and joint3
    for i in range(100):
        joint2_angle = np.linspace(joint2_limits[0], joint2_limits[1], num=100)[i]
        joint3_angle = np.linspace(joint3_limits[0], joint3_limits[1], num=100)[i]
        T2 = dh_transform(a[1], d[1], alpha[1], joint2_angle*joint2_order)
        T3 = dh_transform(a[2], d[2], alpha[2], joint3_angle*joint3_order)

    # Calculate the transformation to the wrist center.
        T = np.linalg.multi_dot([T1, T2, T3])

        wrist_positions.append(T[:-1, -1])  # store only the translation part
        joint_angles.append((joint1_angle,joint2_angle,joint3_angle))

    # convert lists to numpy arrays for convenience
    wrist_positions = np.array(wrist_positions)
    joint_angles = np.array(joint_angles)
    
    return joint_angles, wrist_positions




# Define robot
_a = [0, 0.1, 0.07,0,0,0]
_alpha = [np.pi/2,0,0, np.pi/2, -np.pi/2,0]
_d = [0, 0, 0,0,0,0]

#generate azimuth poly model

_azimuth_joint_angles,_azimuth_wrist_centre_positions = move_around(_a,_d,_alpha)

if debug:
	_joint_positions = calculate_joint_positions(_azimuth_joint_angles,_a,_d,_alpha)
	plot_robot_arm(_joint_positions)

# azimuth angle calculation
azimuth_angles = np.arctan2(_azimuth_wrist_centre_positions[:, 1], _azimuth_wrist_centre_positions[:, 0])

# fit a first order polynomial (line)
azimuth_coeffs = np.polyfit(_azimuth_joint_angles[:, 0], azimuth_angles, 1)

# create a function based on the fit
poly = np.poly1d(azimuth_coeffs)

# generate x values for the fitted line
x = _azimuth_joint_angles

# compute the corresponding y values
y = poly(x)

if debug:
	#plot original data
	plt.figure()
	plt.scatter(_azimuth_joint_angles[:, 0], azimuth_angles, label='original data')
	# plot the fit
	plt.plot(x, y, color='red',label='fit: a=%.2f, b=%.2f' % (coeffs[0], coeffs[1]))
	plt.xlabel('joint1 angle')
	plt.ylabel('azimuth angle')
	plt.legend()
	plt.show()
	

#Generate radial model

rad_joint_angles,rad_wrist_centre_positions = stretch_along(_a,_d,_alpha)
if debug:
	_joint_positions = calculate_joint_positions(_joint_angles,_a,_d,_alpha)
	plot_robot_arm(_joint_positions)
	
# Radial distance calculation
radial_distances = np.sqrt((rad_wrist_centre_positions[:, 1])**2 + (rad_wrist_centre_positions[:, 0])**2)

# Fit a second order polynomial for joint1 angles vs radial distances
coeffs_joint1 = np.polyfit(rad_joint_angles[:,1], radial_distances, 3)  # get coefficients
poly_joint1 = np.poly1d(coeffs_joint1)  # create a function based on the fit

# Fit a second order polynomial for joint2 angles vs radial distances
coeffs_joint2 = np.polyfit(rad_joint_angles[:, 2], radial_distances, 3)  # get coefficients
poly_joint2 = np.poly1d(coeffs_joint2)  # create a function based on the fit

# Generate x values for the fitted line for both joints
x_joint1 = rad_joint_angles[:,1]
x_joint2 = rad_joint_angles[:,2]

# Compute the corresponding y values for both joints
y_joint1 = poly_joint1(x_joint1)
y_joint2 = poly_joint2(x_joint2)

if debug:
	# Plot original data and the fit for joint1
	plt.figure()
	plt.scatter(rad_joint_angles[:, 1], radial_distances, label='original data for joint1')
	plt.plot(x_joint1, y_joint1, color='red', label='fit for joint1: a=%.2f, b=%.2f, c=%.2f, d=%.2f' % tuple(coeffs_joint1))
	plt.xlabel('joint1 angle')
	plt.ylabel('radial distance')
	plt.legend()
	ax = plt.gca()
	plt.show()
	
	#Plot original data and the fit for joint2
	plt.figure()
	plt.scatter(rad_joint_angles[:, 2], radial_distances, label='original data for joint2')
	plt.plot(x_joint2, y_joint2, color='blue', label='fit for joint2: a=%.2f, b=%.2f, c=%.2f, d=%.2f' % tuple(coeffs_joint2))
	plt.xlabel('joint2 angle')
	plt.ylabel('radial distance')
	plt.legend()
	plt.show()
	
#Generate elevation model
# joint angles and wrist centre position for elevation
elevation_joint_angles, elevation_wrist_centre_positions = move_up(_a, _d, _alpha)

if debug:
	_joint_positions = calculate_joint_positions(_joint_angles,_a,_d,_alpha)
	plot_robot_arm(_joint_positions)
	

# compute elevation angles
sqrt_part = np.sqrt(elevation_wrist_centre_positions[:,0]**2 + elevation_wrist_centre_positions[:,1]**2)
offset = 1e-8
elevation_angles = np.arctan2(elevation_wrist_centre_positions[:,2], sqrt_part + offset)


# fit a first order polynomial (line)
elevation_coeffs = np.polyfit(elevation_joint_angles[:,1], elevation_angles, 1)

# create a function based on the fit
poly = np.poly1d(elevation_coeffs)

# generate x values for the fitted line
x = elevation_joint_angles[:,1]

# compute the corresponding y values
y = poly(x)

if debug:
	# plot original data
	plt.figure()
	plt.scatter(elevation_joint_angles[:, 1], elevation_angles, label='original data')
	
	# plot the fit
	plt.plot(x, y, color='red', label='fit: a=%.2f, b=%.2f' % (elevation_coeffs[0], elevation_coeffs[1]))
	plt.xlabel('joint1 angle')
	plt.ylabel('elevation angle')
	plt.legend()
	plt.show()
	

print(azimuth_coeffs, elevation_coeffs,coeffs_joint1,coeffs_joint2)
