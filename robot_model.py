import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from math import asin, atan2, cos, pi

class Robot:
    def __init__(self):
        #6 DOF spherical wrist robot DH params, link length numbers must not be <1. adjust units accordingly.
        self.a = [4,4,4, 0,0,0]
        self.alpha = [np.pi/2,0,0, np.pi/2, 0, np.pi/2]
        self.d = [0, 0, 0, 0,1,1]
        self.joint_offsets = [0,0,0,0,0,0]
        self.joint_limits = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
        self.azimuth_coeffs = None
        self.elevation_coeffs = None
        self.rad_coeffs_joint1=None
        self.rad_coeffs_joint2 = None
    
    def forward_kinematics(self, joint_angle,joint_indx):
        # Calculate each Joint Transformation matrix
        T1 = self.dh_transform(0,joint_angle[0]+self.joint_offsets[0])  
        T2 = self.dh_transform(1,joint_angle[1]+self.joint_offsets[1])
        T3 = self.dh_transform(2,joint_angle[2]+self.joint_offsets[2])
        T4 = self.dh_transform(3,joint_angle[3]+self.joint_offsets[3])  
        T5 = self.dh_transform( 4,joint_angle[4]+self.joint_offsets[4])
        T6 = self.dh_transform(5,joint_angle[5]+self.joint_offsets[5])

        joint_transforms = [T1, T2, T3, T4, T5, T6]
        joint_positions = []
        joint_orientations = []
        accumulated_transform = np.eye(4)
        
        for transform in joint_transforms:
            accumulated_transform = np.dot(accumulated_transform, transform)
            joint_positions.append(accumulated_transform[:3,3])
            
            # Extract the rotation matrix and convert to Euler angles
            rotation_matrix = accumulated_transform[:3,:3]
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('zyx', degrees=False)
            joint_orientations.append(euler_angles)
        
        return np.concatenate((joint_positions[joint_indx], joint_orientations[joint_indx]))
          
    def dh_transform(self,joint_indx,theta):
            a = self.a[joint_indx]
            alpha = self.alpha[joint_indx]
            d = self.d[joint_indx]
            return np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]])
        
    def calculate_joint_positions(self,joint_angles):
        joint_positions = []
        if all(isinstance(angle, (int, float)) for angle in joint_angles): 
          joint_angles = [joint_angles]
        for joint_angle in joint_angles:
            # Calculate each Joint Transformation matrix
            T1 = self.dh_transform(0,joint_angle[0]+self.joint_offsets[0])  
            T2 = self.dh_transform(1,joint_angle[1]+self.joint_offsets[1])
            T3 = self.dh_transform(2,joint_angle[2]+self.joint_offsets[2])
            T4 = self.dh_transform(3,joint_angle[3]+self.joint_offsets[3])  
            T5 = self.dh_transform( 4,joint_angle[4]+self.joint_offsets[4])
            T6 = self.dh_transform(5,joint_angle[5]+self.joint_offsets[5])
        
            # Calculate joint positions        
            Joint_1 = T1[:3,3]
            Joint_2 = np.dot(T1, T2)[:3,3]
            Joint_3 = np.dot(np.dot(T1, T2), T3)[:3,3]
            Joint_4 = np.dot(np.dot(np.dot(T1, T2), T3), T4)[:3,3]
            Joint_5 = np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5)[:3,3]
            # End effector position
            end_effector_position = np.dot(np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5), T6)[:3,3]
            joint_positions.append((Joint_1, Joint_2,Joint_3,Joint_4,Joint_5, end_effector_position))
    
        # Convert joint_positions to a numpy array
        joint_positions_np = np.array(joint_positions)
    
        # Set a threshold
        threshold = 1.745e-6
    
        # Set values below threshold to zero
        joint_positions_np[np.abs(joint_positions_np) < threshold] = 0 
        return joint_positions
        
    def move_around(self):
        wrist_center_positions = []
        joint_angles = []
        theta2 = 0 
        theta3 = 0
        for theta in np.linspace(0, np.pi, num=100):
            theta1 = theta
            T1 = self.dh_transform(0, theta1)
            T2 = self.dh_transform( 1,theta2)
            T3 = self.dh_transform( 2, theta3)
            T = np.linalg.multi_dot([T1, T2, T3])
            wrist_center_positions.append(T[:-1, -1])
            joint_angles.append((theta1, theta2, theta3,0,0,0))
        return np.array(joint_angles), np.array(wrist_center_positions)
    
    def move_up(self):
        wrist_center_positions = []
        joint_angles = []
        theta1 = 0 
        theta3 = 0
        joint2_order = 1 if self.a[1] * self.a[2] > 0 else -1
        for theta in np.linspace(0, np.pi/2, num=100):
            theta2 = theta
            T1 = self.dh_transform( 0,theta1)
            T2 = self.dh_transform( 1,theta2 * joint2_order)
            T3 = self.dh_transform( 2, theta3)        
            T = np.linalg.multi_dot([T1, T2, T3])
            wrist_center_positions.append(T[:-1, -1])
            joint_angles.append((theta1, theta2, theta3,0,0,0))
        return np.array(joint_angles), np.array(wrist_center_positions)
    
        
    def stretch_along(self):
        # Calculate the joint angles
        L2, L3 = self.a[1], self.a[2]  # Lengths of second and third arm segments
        WC_Offset = 0 #self.d[3]  # The offset between third joint and wrist centre point
        x_start = L2 + L3  # starting x position (fully stretched out)
        x_min = abs(L2-L3)
        z = 0  # starting z position
        y = 0  # starting y position (can alter as per requirement)
        
        joint_angles = []
        wrist_positions = []
        for i in range(100):
            xi = np.linspace(x_start, x_min, 100)[i]
            zi = z  # or change as per requirement
            yi = y  # or change as per requirement
            # Account for the wrist centre offset
            d = np.sqrt(xi**2 + zi**2)
            cos_theta3 = np.clip((d**2 - L2**2 - L3**2) / (2 * L2 * L3), -1, 1)
            theta3 = -np.arccos(cos_theta3)
            theta2 = np.arctan2(zi, xi) - np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
            joint_angles.append((0, theta2, theta3, 0, 0, 0))  # add zeros for inactive joints 1, 4, 5, 6
            wrist_positions.append((xi, yi- WC_Offset, zi))  # WC_Offset is added to y to get the position of the wrist center
        return np.array(joint_angles), np.array(wrist_positions)

    def calculate_R0_3(self,theta_1, theta_2, theta_3):
        # Compute individual transformation matrices
    
        T0_1 = self.dh_transform(0,theta_1)  
        T1_2 = self.dh_transform(1, theta_2) # Flat position
        T2_3 = self.dh_transform(2, theta_3) # Flat position
        # Multiply the matrices together to get the transformation matrix from frame 0 to frame 3
        T0_2 = np.dot(T0_1, T1_2)
        T0_3 = np.dot(T0_2, T2_3)
    
        # Extract rotation matrix R0_3 from the transformation matrix
        R0_3 = T0_3[0:3, 0:3]
    
        return R0_3
    def solve_orientation(self,R36):
            theta_5 = np.arctan2(np.sqrt(R36[0,2]**2+ R36[1, 2]**2),- R36[2, 2])
            theta_4 = np.arctan2(-R36[1, 2], -R36[0, 2])
            theta_6 = np.arctan2(R36[2, 1], R36[2, 0])
            return theta_4,theta_5,theta_6
            #the5_R.append(np.arctan2(-np.sqrt(1 - R36[j][1, 2]**2), R36[j][1, 2]))
#            the4_R.append(np.arctan2(-R36[j][2, 2], R36[j][0, 2]))
#            the6_R.append(np.arctan2(R36[j][1, 1], -R36[j][1, 0]))
       
    def calculate_last_three_joints(self,theta_1, theta_2, theta_3, roll, pitch, yaw,ax=None,wrist_loc = None):

        # Calculate rotation matrix R0_3 using first three joint angles
        R0_3 = self.calculate_R0_3(theta_1, theta_2, theta_3)

        # Calculate and return last three joint angles
        # Calculate target orientation in the wrist frame
        R0_6 = R.from_euler('zyx', [roll,pitch,yaw], degrees=False).as_matrix()
        R3_6 = np.dot(np.transpose(R0_3), R0_6)
        theta_4,theta_5,theta_6 = self.solve_orientation(R3_6)
        #theta_4,theta_5,theta_6 = R.from_matrix(R3_6).as_euler('zyx', degrees=False)
        
        if ax:
            R0_3_vector = R0_3[:, 0] 
            R_3_6_vector = R3_6[:, 0]
            target_rotation_vector = R0_6[:, 0]
            
            ax.quiver(wrist_loc[0], wrist_loc[1], wrist_loc[2], R0_3_vector[0], R0_3_vector[1], R0_3_vector[2], color='b')
            ax.quiver(wrist_loc[0], wrist_loc[1], wrist_loc[2], R_3_6_vector[0], R_3_6_vector[1], R_3_6_vector[2], color='r')
            ax.quiver(wrist_loc[0], wrist_loc[1], wrist_loc[2], target_rotation_vector[0], target_rotation_vector[1], target_rotation_vector[2], color='g')
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Rotation Vectors')
    
        return (theta_4, theta_5, theta_6)
    
        
    def wrist_center_position(self, target_position, target_orientation):
        R_matrix = R.from_euler('zyx', target_orientation,degrees=False).as_matrix()
        a4 = self.a[-1]
        displacement = np.array([a4, 0, 0])
        
        # Transform the displacement to the end effector coordinate frame
        transformation = R_matrix @ displacement
        
        # Subtract the transformation from the end effector's position to get wrist center 
        wrist_center = target_position + transformation
          
        return wrist_center

