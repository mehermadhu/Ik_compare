from kinematics_solution_finder import my_IK
from robot_model import Robot
import coordinate_conversions as utils
import visualization as viz
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from memory_profiler import memory_usage

show_robot  = False
show_plot = False #'azimuth', 'elevation', 'radius1', 'radius2'
debug = True
show_ik = False
compute_complexity = True
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# Define robot
manipulator = Robot()
#irb 120
manipulator.a = [0,4,4, 0, 0, 1] 
manipulator.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2,0] 
manipulator.d = [0, 0, 0, 0,0,0]
manipulator.joint_offsets = [0,0,0,0,0,0]


#manipulator.a = [0,-0.425,-0.3922,0,0,0] 
#manipulator.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0] 
#manipulator.d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
#generate azimuth model
calib_start_time = time.time()
calib_before_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
_azimuth_joint_angles,_azimuth_wrist_centre_positions = manipulator.move_around()

if show_robot:
	_joint_positions = manipulator.calculate_joint_positions(_azimuth_joint_angles)
	viz.plot_robot_arm(_joint_positions,ax)

# azimuth angle calculation
azimuth_angles = utils.azimuth_angle_calculation(_azimuth_wrist_centre_positions)

# fit a first order polynomial (line)
azimuth_coeffs = np.polyfit(azimuth_angles, _azimuth_joint_angles[:, 0], 1)

# create a function based on the fit
poly = np.poly1d(azimuth_coeffs)

# generate x values for the fitted line
x =azimuth_angles

# compute the corresponding y values
y = poly(x)

if show_plot == 'azimuth':
	viz.plot_data(azimuth_angles, _azimuth_joint_angles[:, 0], x,y,'azimuth','base_angle')
	

#Generate radial model

rad_joint_angles,rad_wrist_centre_positions = manipulator.stretch_along()
	       
# Radial distance calculation
radial_distances = utils.radial_distance_calculation(rad_wrist_centre_positions)

# Fit a second order polynomial for joint1 angles vs radial distances
rad_coeffs_joint1 = np.polyfit( radial_distances,rad_joint_angles[:,1], 12)  # get coefficients
poly_joint1 = np.poly1d(rad_coeffs_joint1)  # create a function based on the fit

# Fit a second order polynomial for joint2 angles vs radial distances
rad_coeffs_joint2 = np.polyfit(radial_distances,rad_joint_angles[:, 2],  12)  # get coefficients
poly_joint2 = np.poly1d(rad_coeffs_joint2)  # create a function based on the fit

# Generate x values for the fitted line for both joints
x_joint1 = radial_distances
x_joint2 = radial_distances

# Compute the corresponding y values for both joints
y_joint1 = poly_joint1(x_joint1)
y_joint2 = poly_joint2(x_joint2)

if show_plot == 'radius1':   
	# Plot original data and the fit for joint1
	viz.plot_data(radial_distances,rad_joint_angles[:, 1], x_joint1, y_joint1,'shoulder_angle','radius')
if show_plot == 'radius2':
	#Plot original data and the fit for joint2
	viz.plot_data(radial_distances,rad_joint_angles[:,2], x_joint2, y_joint2,'shoulder_angle','radius')
	
#Generate elevation model
# joint angles and wrist centre position for elevation
elevation_joint_angles, elevation_wrist_centre_positions = manipulator.move_up()

if show_robot:
	_joint_positions = manipulator.calculate_joint_positions(rad_joint_angles)
	viz.plot_robot_arm(_joint_positions,ax)
	

# compute elevation angles
elevation_angles = utils.elevation_angle_calculation(elevation_wrist_centre_positions)

# fit a first order polynomial (line)
elevation_coeffs = np.polyfit(elevation_joint_angles[:,1], elevation_angles, 1)

# create a function based on the fit
poly = np.poly1d(elevation_coeffs)

# generate x values for the fitted line
x = elevation_joint_angles[:,1]

# compute the corresponding y values
y = poly(x)

if show_plot == 'elevation':
	# plot original data
	viz.plot_data(elevation_joint_angles[:, 1], elevation_angles,x,y,'shoulder_angle','elevation')
if show_robot:
	_joint_positions = manipulator.calculate_joint_positions(elevation_joint_angles)
	viz.plot_robot_arm(_joint_positions,ax)    
# time stamp
calib_after_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
calib_end_time = time.time()
#Now you can test the inverse kinematics function with your data.
curr_joint_angles = [0, 0, 0, 0, 0, 0]  # Example current joint angles

target_positions = [
    np.array([4,0,0])
]
target_orientations = [np.array([0,np.pi/2,0])]
max_pos = 4
min_pos = 0

# generate target position
if debug:
    target_positions = []
    target_orientations = []
    for i in range(10):
        target_positions.append([random.uniform(min_pos, max_pos) for _ in range(3)])
        # generate target Euler angles for orientation (roll, pitch, yaw) 
        #target_orientations.append([random.uniform(-np.pi, np.pi) for _ in range(3)])\
        target_orientations.append([0,np.pi/2,0])
manipulator.azimuth_coeffs = azimuth_coeffs
manipulator.rad_coeffs_joint1 = rad_coeffs_joint1
manipulator.rad_coeffs_joint2 = rad_coeffs_joint2
manipulator.elevation_coeffs = elevation_coeffs

target_idx = 0
if compute_complexity:
    solve_start_time = time.time()
    solve_before_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
    
for target_pos, target_orient in zip(target_positions, target_orientations):
    wrist_target_pos = manipulator.wrist_center_position(target_pos,target_orient)
    joint_angles = my_IK(manipulator, wrist_target_pos, target_orient, curr_joint_angles,ax)
    if not joint_angles is None:
        curr_joint_angles = joint_angles
        joint_positions = manipulator.calculate_joint_positions(joint_angles)
        ee_pos = joint_positions[-1][-1]
        wrist_pos = joint_positions[-1][2]
        tolerance = 0.1  # Adjust depending on the precision of your system
        try:
            assert np.allclose(target_pos, ee_pos, atol=tolerance), \
            f'Target position: {target_pos} not reached by end effector: {ee_pos}'
        except AssertionError as e:
            print(e)
            #print(f'target wrist pose: {wrist_target_pos} actual wrist pose: {wrist_pos}')
        else:
            print(f'Target position: {target_pos} reached by end effector: {ee_pos}')
            #print(f'target wrist pose: {wrist_target_pos} actual wrist pose: {wrist_pos}')
        if show_ik:
            viz.plot_robot_arm(joint_positions,ax)
if compute_complexity:
    solve_after_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
    solve_end_time = time.time()
    print(f"Execution Time to Solve: {(solve_end_time - solve_start_time)/10} seconds")
    print(f"Memory used to solve: {solve_after_memory - solve_before_memory} MiB")
    print(f"Execution Time to calibrate: {calib_end_time - calib_start_time} seconds")
    print(f"Memory used to calibrate: {calib_after_memory - calib_before_memory} MiB")
if show_ik:
    plt.show()