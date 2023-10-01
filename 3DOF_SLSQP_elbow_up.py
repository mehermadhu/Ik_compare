import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dh_transform_matrix(a, alpha, d, theta):
    A = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return A

a = [8, 8, 8, 8, 8, 8]
alpha = [np.radians(0), np.radians(90), np.radians(90), np.radians(90), np.radians(90), np.radians(90)]
d = [1, 1, 1, 1, 1, 1]

def forward_kinematics(theta):
    A = [dh_transform_matrix(a[i], alpha[i], d[i], theta[i]) for i in range(6)]
    T = np.array([A[0]])
    for i in range(1, 6):
        T = np.append(T, [np.dot(T[i-1], A[i])], axis=0)
    return T

def objective(theta, target_pos, target_orient):
    T = forward_kinematics(theta)
    position_error = np.sum(np.square(T[-1][:3, 3] - target_pos))
    orientation_error = np.sum(np.square(T[-1][:3, :3] - target_orient))
    return position_error + orientation_error

con = [{'type': 'ineq', 'fun': lambda x: np.pi - np.abs(x[i])} for i in range(6)]
con.append({'type': 'ineq', 'fun': lambda x: forward_kinematics(x)[1][2,3]})

target_positions = [
    np.array([5, 5, 15]),
    np.array([10, 10, 0]),
    np.array([15, 10, -10]),
    np.array([5, -10, 5]),
]
target_orientations = [np.identity(3) for _ in range(4)] 
x0 = [0, 0, 0, 0, 0, 0] 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-15, 15)

for target_pos, target_orient in zip(target_positions, target_orientations):
    res = minimize(objective, x0, args=(target_pos, target_orient), constraints=con, method='SLSQP')
    T = forward_kinematics(res.x)
    
    ax.plot([0, T[0][0,3]], [0, T[0][1,3]], zs=[0, T[0][2,3]], color='red')
    for i in range(2, 7):  # Adjust the loop range and indexing of T
        ax.plot([T[i-2][0,3], T[i-1][0,3]], [T[i-2][1,3], T[i-1][1,3]], zs=[T[i-2][2,3], T[i-1][2,3]], color='blue' if i%2 else 'black')
    ax.plot([T[5][0,3], T[5][0,3]], [T[5][1,3], T[5][1,3]], zs=[T[5][2,3], T[5][2,3]], color='green')
 
    plt.pause(0.5)

plt.show()
