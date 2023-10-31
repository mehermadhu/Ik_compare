import numpy as np
from scipy.spatial.transform import Rotation

# Your Robotic Arm configs
DH_table = { 'alpha': [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
             'a': [0, 0.425, 0.3922, 0, 0, 0],
             'd': [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]} #

class UR5e:
    def __init__(self, DH_table):
        # dh parameters
        self.a, self.alpha, self.d = [list(val) for val in DH_table.values()]
        self.d_wc = self.d[-1]  # Distance from the wrist center to the end effector
        self.DH_table = DH_table

    def dh_transform_matrix(self, alpha, a, d, theta):
        A = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return A
    def calculate_joint_angles(self, target_position, target_orientation):
        # Find wrist center
        target_rotation_matrix = Rotation.from_euler('xyz', target_orientation).as_matrix()
        wrist_center = target_position - self.d_wc * target_rotation_matrix[:, 2]
        # Geometric IK for spherical wrist robot
        theta1 = np.arctan2(wrist_center[1], wrist_center[0])
        r = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2)
        s = wrist_center[2] - self.d[0]

        D = (r**2 + s**2 - np.array(self.a[1])**2 - np.array(self.a[2])**2) / (2*np.array(self.a[1])*np.array(self.a[2]))
        D = np.clip(D, -1, 1)  # This ensures D stays within [-1, 1]
        theta3 = np.arctan2(np.sqrt(1 - D**2), D)
        
        alpha = np.arctan2(s, r)
        beta = np.arctan2(np.array(self.a[2])*np.sin(theta3), np.array(self.a[1]) + np.array(self.a[2])*np.cos(theta3))
        theta2 = alpha - beta
        
        # Use the target rotation matrix for your orientation IK
        rotation_matrix_0_3 = np.dot(np.dot(self.dh_transform_matrix(self.alpha[0],self.a[0],self.d[0],theta1),
                              self.dh_transform_matrix(self.alpha[1],self.a[1],self.d[1],theta2)),
                              self.dh_transform_matrix(self.alpha[2],self.a[2],self.d[2],theta3))
        
        rotation_matrix_0_3 = rotation_matrix_0_3[:3, :3]
        rotation_matrix_0_3_inv = np.linalg.inv(rotation_matrix_0_3)
        rotation_matrix_3_6 = np.dot(rotation_matrix_0_3_inv, target_rotation_matrix)
        
        theta4 = np.arctan2(rotation_matrix_3_6[1, 2], rotation_matrix_3_6[0, 2])
        theta5 = np.arctan2(np.sqrt(rotation_matrix_3_6[0, 2]**2 + rotation_matrix_3_6[1, 2]**2), rotation_matrix_3_6[2, 2])
        theta6 = np.arctan2(rotation_matrix_3_6[2, 1], -rotation_matrix_3_6[2, 0])
        
        return [theta1, theta2, theta3, theta4, theta5, theta6]
        
    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        for i, angle in enumerate(joint_angles):
            alpha, a, d = [float(val[i]) for val in self.DH_table.values()]
            angle = float(angle)  # convert sympy.Float to float
            A_i = np.array([[np.cos(angle),    -np.sin(angle)*np.cos(alpha),     np.sin(angle)*np.sin(alpha),     a*np.cos(angle)],
                   [np.sin(angle),    np.cos(angle)*np.cos(alpha),     -np.cos(angle)*np.sin(alpha),     a*np.sin(angle)],
                   [0,              np.sin(alpha),                 np.cos(alpha),                 d],
                   [0,              0,                            0,                         1]])
            T = np.dot(T, A_i)
        T = np.round(T, decimals=4)
        position = T[:3, 3]
        R = T[:3, :3]
        alpha = np.arctan2(R[2, 1], R[2, 2])
        beta = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        gamma = np.arctan2(R[1, 0], R[0, 0])
        orientation = np.array([alpha, beta, gamma])
        return position, orientation
    
manipulator = UR5e(DH_table)

target_position = np.array([0.1, 0.2, 0.3])
target_orientation = np.array([np.deg2rad(90),0,0])
joint_angles = manipulator.calculate_joint_angles(target_position, target_orientation)

fk_position, fk_orientation = manipulator.forward_kinematics(joint_angles)

position_error = np.sqrt(np.sum((fk_position - target_position)**2))
orientation_error = np.sqrt(np.sum((fk_orientation - target_orientation)**2))

print('Joint angles: ', joint_angles)
print('FK position: ', fk_position)
print('FK orientation: ', fk_orientation)
print('Position error: ', position_error)
print('Orientation error: ', orientation_error)