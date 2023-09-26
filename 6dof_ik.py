import numpy as np
from numpy import sin, cos, sqrt, arctan2
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Data generation for circular and straight line path
import matplotlib.pyplot as plt
import numpy as np
import math


# Assume these values as DH parameters of KUKA R650
a1, a2, a3 = 0, 250, 700
α1, α2, α3 = 0, np.pi / 2 , 0
d1, d2, d3 = 400, 0, 0
   
import numpy as np

# Define lengths of segments based on the Denavit-Hartenberg (DH) parameters.
d1 = 400  # Length from base to shoulder.
a2 = 250  # Length from shoulder to elbow.
a3 = 700  # Length from elbow to wrist.


def calculate_position(base_angle, shoulder_angle, elbow_angle):
    x = d1*np.cos(base_angle) + a2*np.cos(shoulder_angle) + a3*np.cos(elbow_angle) # x-coordinate
    y = d1*np.sin(base_angle) + a2*np.sin(shoulder_angle) + a3*np.sin(elbow_angle) # y-coordinate
    z = d1 + a2*np.cos(shoulder_angle) + a3*np.cos(elbow_angle)  # z-coordinate
    return x, y, z


def calculate_elevation():
    theta_values = np.linspace(0, np.pi, num=1000)  # Range of shoulder joint rotation angles

    theta_phi_list = []  # List to store tuple of joint angle and elevation angle

    for theta in theta_values:
        x, y, z = calculate_position(0, theta, theta)

        xy_distance = np.sqrt(x**2 + y**2)
        elevation_angle = np.arctan2(z, xy_distance)

        theta_phi_list.append((theta, elevation_angle))

    return theta_phi_list


def calculate_azimuth(base_rotation_values):
    base_azimuth_list = []  # List to store tuple of base joint angle and azimuth angle

    for base_rotation in base_rotation_values:
        x, y, z = calculate_position(base_rotation, 0, 0)
        
        azimuth = np.arctan2(y, x)
        base_azimuth_list.append((base_rotation, azimuth))

    return base_azimuth_list


base_rotation_values = np.linspace(0, 2*np.pi, num=1000)
azimuth_data = calculate_azimuth(base_rotation_values)
elevation_data = calculate_elevation()

# Extract the joint angles and azimuth/elevation angles from the data
base_angles, azimuths = zip(*azimuth_data)
shoulder_angles, elevations = zip(*elevation_data)

# Fit first-order polynomials
base_poly_coeff = np.polyfit(azimuths, base_angles, 1)
shoulder_poly_coeff = np.polyfit(elevations, shoulder_angles, 1)

# We now have the coefficients of the first-order polynomials
# base_angle = base_poly_coeff[0]*azimuth + base_poly_coeff[1]
# shoulder_angle = shoulder_poly_coeff[0]*elevation + shoulder_poly_coeff[1]


def calculate_base_angle(azimuth):
    return base_poly_coeff[0]*azimuth + base_poly_coeff[1]


def calculate_shoulder_angle(elevation):
    return shoulder_poly_coeff[0]*elevation + shoulder_poly_coeff[1]

#----------------------------------updated sofar---------------------------


def find_best_solution(poly, value, joint_limits, curr_joint_angle):
    roots = np.roots(poly - value)
    real_roots = roots[~np.iscomplex(roots)].real

    if len(real_roots) > 0:
        real_roots_min, real_roots_max = np.min(real_roots), np.max(real_roots)
        joint_min, joint_max = joint_limits

        if real_roots_min >= joint_min and real_roots_max <= joint_max:
            solution = real_roots_min if abs(real_roots_min - curr_joint_angle) < abs(real_roots_max - curr_joint_angle) else real_roots_max
        elif real_roots_min >= joint_min:
            solution = real_roots_min
        elif real_roots_max <= joint_max:
            solution = real_roots_max
        else:
            print("No valid joint solution found within the joint limits.")
            return None
    else:
        print("No real roots were obtained from the polynomial.")
        return None

    return solution


def my_IK(x, y, curr_joint_angles, joint_limits):
    angle = np.arctan2(y, x)  # Calculate azimuth
    radius = np.sqrt(x**2 + y**2)  # Calculate radius

    # Predict delta_theta1 and theta2 using polynomial models and find best solution
    joint1_azimuth = find_best_solution(polynomial_joint1_azimuth, angle, joint_limits[0], curr_joint_angles[0])
    delta_joint1_radii = find_best_solution(polynomial_joint1_radii, radius, joint_limits[0], curr_joint_angles[0])
    joint2_radii = find_best_solution(polynomial_joint2_radii, radius, joint_limits[1], curr_joint_angles[1])

    if joint1_azimuth is None or delta_joint1_radii is None or joint2_radii is None:
        return None, None
    
    # Calculate joint angles
    joint1_angle = joint1_azimuth #+ delta_joint1_radii
    joint2_angle = joint2_radii

    return joint1_angle, joint2_angle


# Define helper function to form individual transformation matrices
def TF_Matrix(alpha, a, d, theta):
    TF = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                   [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                   [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                   [0, 0, 0, 1]])
    return TF

def calculate_R0_3(theta_1, theta_2, theta_3):
    # Compute individual transformation matrices
    T0_1 = TF_Matrix(α1, a1, d1, theta_1)
    T1_2 = TF_Matrix(α2, a2, d2, theta_2)
    T2_3 = TF_Matrix(α3, a3, d3, theta_3)

    # Multiply the matrices together to get the transformation matrix from frame 0 to frame 3
    T0_2 = np.dot(T0_1, T1_2)
    T0_3 = np.dot(T0_2, T2_3)

    # Extract rotation matrix R0_3 from the transformation matrix
    R0_3 = T0_3[0:3, 0:3]

    return R0_3


def calculate_joint_angles(R3_6):
    theta_4 = np.arctan2(R3_6[1, 2], -R3_6[0, 2])
    theta_5 = np.arctan2(np.sqrt(R3_6[0, 2] ** 2 + R3_6[1, 2] ** 2), R3_6[2, 2])
    theta_6 = np.arctan2(-R3_6[2, 1], R3_6[2, 0])

    return (theta_4, theta_5, theta_6)

def calculate_R3_6(roll, pitch, yaw, R0_3):
    # Calculate rotation matrix R0_G from roll, pitch, yaw
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R0_G = np.dot(R_z, np.dot(R_y, R_x))

    # Calculate R3_6
    R3_6 = np.dot(np.transpose(R0_3), R0_G)

    return R3_6


def calculate_last_three_joints(theta_1, theta_2, theta_3, roll, pitch, yaw):
    
    # Calculate rotation matrix R0_3 using first three joint angles
    R0_3 = calculate_R0_3(theta_1, theta_2, theta_3)
    
    # Calculate rotation matrix R3_6 using roll, pitch, yaw, and R0_3
    R3_6 = calculate_R3_6(roll, pitch, yaw, R0_3)
    
    # Calculate and return last three joint angles
    theta_4, theta_5, theta_6 = calculate_joint_angles(R3_6)
    
    return (theta_4, theta_5, theta_6)



# Step-1
# Generating data
joint_angles_straight_line, radii_straight_line = move_in_straight_line()
joint_angles_circular_path, azimuths_circular_path = move_in_circular_path(a1+a2)  

# Step-2 Plotting data
# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Prepare data for polynomial fits
joint1_angles_straight_line = [angles[0] for angles in joint_angles_straight_line]
joint2_angles_straight_line = [angles[1] for angles in joint_angles_straight_line]
joint1_angles_circular_path = [angles[0] for angles in joint_angles_circular_path]

# Fit second order polynomial for Joint 1 Angle vs Radii - Straight line
polynomial_joint1_radii = np.polyfit(joint1_angles_straight_line, radii_straight_line, 2)
polynomial_values_joint1_radii = np.polyval(polynomial_joint1_radii, joint1_angles_straight_line)

axs[0, 0].plot(joint1_angles_straight_line, radii_straight_line)
axs[0, 0].plot(joint1_angles_straight_line, polynomial_values_joint1_radii, 'r-')

# Fit second order polynomial for Joint 2 Angle vs Radii - Straight line
polynomial_joint2_radii = np.polyfit(joint2_angles_straight_line, radii_straight_line, 2)
polynomial_values_joint2_radii = np.polyval(polynomial_joint2_radii, joint2_angles_straight_line)

axs[0, 1].plot(joint2_angles_straight_line, radii_straight_line)
axs[0, 1].plot(joint2_angles_straight_line, polynomial_values_joint2_radii, 'r-')

# Fit first order polynomial for Joint 1 Angle vs Azimuth - Circular path
polynomial_joint1_azimuth = np.polyfit(joint1_angles_circular_path, azimuths_circular_path, 1)
polynomial_values_joint1_azimuth = np.polyval(polynomial_joint1_azimuth, joint1_angles_circular_path)

axs[1, 0].plot(joint1_angles_circular_path, azimuths_circular_path)
axs[1, 0].plot(joint1_angles_circular_path, polynomial_values_joint1_azimuth, 'r-')

# Set titles, labels, etc.
# *** omitted for brevity ***

# Show plots
plt.show()

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Constants
a1 = a2 = 1  # Equal link lengths
N = 100 # number of samples for each joint angle

# Step 1 - Generate Training Data
theta1_range = np.linspace(-np.pi, np.pi, N) 
theta2_range = np.linspace(-np.pi, np.pi, N)
theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)

theta1_values = theta1_grid.flatten()
theta2_values = theta2_grid.flatten()

x_values = a1 * np.cos(theta1_values) + a2 * np.cos(theta1_values + theta2_values)
y_values = a1 * np.sin(theta1_values) + a2 * np.sin(theta1_values + theta2_values)

inputs = np.array([x_values, y_values]).T
outputs = np.array([theta1_values, theta2_values]).T

# Step 2 - Define and Train ANN model
model = Sequential([  
    Dense(32, input_dim=2, activation='relu'),
    Dense(2, activation='linear')
])

model.compile(loss='mse', optimizer='adam')
model.fit(inputs, outputs, epochs=200, batch_size=32, validation_split=0.2)


# Step 3 - Test ANN model
test_theta1 = np.random.uniform(-np.pi/2, np.pi/2, 10) # Values for Joint1 lies between -pi/2 to pi/2 
test_theta2 = np.random.uniform(-np.pi, np.pi, 10) # Random values for Joint2 between -pi and pi
test_points = np.column_stack((test_theta1, test_theta2))

# Converting joint angles to cartesian coordinates
test_points_cartesian = np.zeros_like(test_points)
test_points_cartesian[:,0] = a1 * np.cos(test_theta1) + a2 * np.cos(test_theta1 + test_theta2)
test_points_cartesian[:,1] = a1 * np.sin(test_theta1) + a2 * np.sin(test_theta1 + test_theta2)

ann_pred_angles = model.predict(test_points_cartesian)
poly_pred_angles = [np.array(my_IK(x, y, [0,0], [(-np.pi/2, np.pi/2), (-np.pi, np.pi)])) for x, y in test_points_cartesian]

# Step 4 - Compare performance
ann_mse = mean_squared_error(test_points, ann_pred_angles)
poly_mse = mean_squared_error(test_points, poly_pred_angles)
print("ANN MSE: ", ann_mse)
print("Polynomial MSE: ", poly_mse)
