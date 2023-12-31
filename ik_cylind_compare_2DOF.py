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

# Constants
a1 = a2 = 1  # Equal link lengths

fig, ax = plt.subplots()  # Preparing the plot

ax.set_xlim([-2,2]) #Setting x-axis limits
ax.set_ylim([-2,2]) #Setting y-axis limits
line, = ax.plot([],[]) #Creating a line object to be updated

# Draw Function
def draw(theta1, theta2):
    x = [0, a1*np.cos(theta1), a1*np.cos(theta1) + a2*np.cos(theta1 + theta2)]
    y = [0, a1*np.sin(theta1), a1*np.sin(theta1) + a2*np.sin(theta1 + theta2)]

    line.set_data(x, y) #Updating the line object
    plt.draw()
    plt.pause(0.01) #Pause for 0.01 seconds

    plt.title('Robot Arm')
    plt.grid(True)

def IK(x, y):
    d = (x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2)

    if abs(d) > 1:
        print("The target is not reachable.")
        return None, None

    theta2 = np.arccos(d) #theta2 is deduced from cos(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(a2 * np.sin(theta2), a1 + a2 * np.cos(theta2)) #theta1 is deduced from sin(theta1)

    return theta1, theta2 #Returning the angles in radians
def move_in_straight_line():
    x_values = np.linspace(0.1, (a1 + a2), num=50)
    y_test = 0
    joint_angles_list = []
    radii = []

    for x_test in x_values:
        theta1_test, theta2_test = IK(x_test, y_test)
        if theta1_test is None:
            continue
        joint_angles_list.append((theta1_test, theta2_test))

        r = np.sqrt(x_test**2 + y_test**2)  # Calculate the radius
        radii.append(r)

        draw(theta1_test, theta2_test)
    
    return joint_angles_list, radii

def move_in_circular_path(radius):
    angle_values = np.linspace(-np.pi/2,np.pi/2, num=100)
    thetas = []
    azimuths = []

    for angle in angle_values:
        theta1 = angle
        theta2 = 0  
        x = radius*np.cos(theta1)
        y = radius*np.sin(theta1)
        thetas.append((theta1, theta2))  

        azimuth = np.arctan2(y, x) # Calculate the azimuth angle
        azimuths.append(azimuth)

        draw(theta1, theta2)
    return thetas, azimuths
    
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

# Fit second order polynomial for Joint 1 Angle vs Radii - Straight line
polynomial_joint1_radii = np.polyfit(joint1_angles_straight_line, radii_straight_line, 2)
polynomial_values_joint1_radii = np.polyval(polynomial_joint1_radii, joint1_angles_straight_line)

joint1_predictions = [pred[0] for pred in poly_pred_angles if pred is not None]
joint1_predicted_values = np.polyval(polynomial_joint1_radii, joint1_predictions)

axs[0, 0].plot(joint1_angles_straight_line, radii_straight_line)
axs[0, 0].scatter(joint1_predictions, joint1_predicted_values, color='green')
axs[0, 0].plot(joint1_angles_straight_line, polynomial_values_joint1_radii, 'r-')

# Fit second order polynomial for Joint 2 Angle vs Radii - Straight line
polynomial_joint2_radii = np.polyfit(joint2_angles_straight_line, radii_straight_line, 2)
polynomial_values_joint2_radii = np.polyval(polynomial_joint2_radii, joint2_angles_straight_line)

joint2_predictions = [pred[1] for pred in poly_pred_angles if pred is not None]
joint2_predicted_values = np.polyval(polynomial_joint2_radii, joint2_predictions)

axs[0, 1].plot(joint2_angles_straight_line, radii_straight_line)
axs[0, 1].scatter(joint2_predictions, joint2_predicted_values, color='green')
axs[0, 1].plot(joint2_angles_straight_line, polynomial_values_joint2_radii, 'r-')

# Fit first order polynomial for Joint 1 Angle vs Azimuth - Circular path
polynomial_joint1_azimuth = np.polyfit(joint1_angles_circular_path, azimuths_circular_path, 1)
polynomial_values_joint1_azimuth = np.polyval(polynomial_joint1_azimuth, joint1_angles_circular_path)

azimuth_predicted_values = np.polyval(polynomial_joint1_azimuth, joint1_predictions)

axs[1, 0].plot(joint1_angles_circular_path, azimuths_circular_path)
axs[1, 0].scatter(joint1_predictions, azimuth_predicted_values, color='green')
axs[1, 0].plot(joint1_angles_circular_path, polynomial_values_joint1_azimuth, 'r-')

# After you configure all other plotting options (like titles, labels, etc.), show the plots
plt.show()