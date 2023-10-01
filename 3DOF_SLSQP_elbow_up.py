# Necessary Imports
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to calculate the Denavit-Hartenberg transformation matrix
def dh_transform_matrix(a, alpha, d, theta):
    A = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
        [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
        [0, 0, 0, 1]
    ])
    return A

# DH Parameters (alpha and a are in degrees, d is in units of length)
a = [8, 8, 8]
alpha = [np.radians(0), np.radians(90), np.radians(90)]
d = [1, 1, 1]

# Function to calculate Forward Kinematics
def forward_kinematics(theta):
    A1 = dh_transform_matrix(a[0], alpha[0], d[0], theta[0])
    A2 = dh_transform_matrix(a[1], alpha[1], d[1], theta[1])
    A3 = dh_transform_matrix(a[2], alpha[2], d[2], theta[2])
    return A1, np.dot(A1, A2), np.dot(A1, np.dot(A2, A3))

# Objective function to be minimized
def objective(theta, target):
    _, _, T3 = forward_kinematics(theta)
    end_effector_pos = T3[:3,3]
    return np.sum(np.square(end_effector_pos - target))

# Constraints for the optimization problem
con = ({'type': 'ineq', 'fun': lambda x: np.pi - np.abs(x[0])},
       {'type': 'ineq', 'fun': lambda x: np.pi - np.abs(x[1])},
       {'type': 'ineq', 'fun': lambda x: np.pi - np.abs(x[2])},
       {'type': 'ineq', 'fun': lambda x: forward_kinematics(x)[1][2,3]})

# Target positions for the end-effector
target_positions = [
    np.array([5, 5, 15]),
    np.array([10, 10, 0]),
    np.array([15, 10, -10]),
    np.array([5, -10, 5]),
]
x0 = [0, 0, 0]  # Initial guess for theta values

# Configure the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-15, 15)

# Optimize and plot the robot's movements for each target position
for target_pos in target_positions:
    res = minimize(objective, x0, args=(target_pos), constraints=con, method='SLSQP')
    
    T1, T2, T3 = forward_kinematics(res.x)

    ax.plot([0, T1[0,3]], [0, T1[1,3]], zs=[0, T1[2,3]], color='red')
    ax.plot([T1[0,3], T2[0,3]], [T1[1,3], T2[1,3]], zs=[T1[2,3], T2[2,3]], color='blue')
    ax.plot([T2[0,3], T3[0,3]], [T2[1,3], T3[1,3]], zs=[T2[2,3], T3[2,3]], color='green')
 
    plt.pause(0.5)  # Add short pause for animation

plt.show()
