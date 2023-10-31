import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def compute_trans_mat(theta, d, a, alpha):
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [0, np.sin(alpha), np.cos(alpha), d],
                  [0, 0, 0, 1]])
    return T
    
DH_params = np.array([[0, 3.52, 0.70, -np.pi/2], 
                      [0, 0, 3.6, 0], 
                      [0, 0, 0, -np.pi/2], 
                      [0, 3.8, 0, np.pi/2],
                      [0, 0, 0, -np.pi/2],
                      [0, 0.65, 0, 0]])

joint_angles = [0,0,0,0,0,0]
joint_angles[1] = -np.pi/2
joint_angles[3] = np.pi/4

T0_6= np.eye(4)

joint_positions = np.zeros((7, 4))
for i in range(6):
    T_i_i1 = compute_trans_mat(joint_angles[i], DH_params[i,1], DH_params[i,2], DH_params[i,3]) 
    T0_6 = np.dot(T0_6, T_i_i1) 
    joint_positions[i+1,:3] = T0_6[:-1, -1]
    joint_positions[i+1,3] = 1 

# Set a threshold
threshold = 1.745e-6
# Set values below threshold to zero
joint_positions[abs(joint_positions) <threshold] = 0

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot(joint_positions[:,0], joint_positions[:,1], joint_positions[:,2])
ax.scatter(joint_positions[:,0], joint_positions[:,1], joint_positions[:,2])
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.set_xlim([np.min(joint_positions),np.max(joint_positions)])
ax.set_ylim([np.min(joint_positions),np.max(joint_positions)])
ax.set_zlim([np.min(joint_positions),np.max(joint_positions)])
plt.show()
