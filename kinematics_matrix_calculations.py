import numpy as np

def dh_transform(a, d, alpha, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]])



def calculate_R0_3(_a,_d,_alpha,theta_1, theta_2, theta_3):
    # Compute individual transformation matrices

    T0_1 = dh_transform(_a[0], _d[0], _alpha[0], theta_1)  
    T1_2 = dh_transform(_a[1], _d[1], _alpha[1], theta_2) # Flat position
    T2_3 = dh_transform(_a[2], _d[2], _alpha[2], theta_3) # Flat position
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

def calculate_last_three_joints(_a,_d,_alpha,theta_1, theta_2, theta_3, roll, pitch, yaw):
    
    # Calculate rotation matrix R0_3 using first three joint angles
    R0_3 = calculate_R0_3(_a,_d,_alpha,theta_1, theta_2, theta_3)
    
    # Calculate rotation matrix R3_6 using roll, pitch, yaw, and R0_3
    R3_6 = calculate_R3_6(roll, pitch, yaw, R0_3)
    
    # Calculate and return last three joint angles
    theta_4, theta_5, theta_6 = calculate_joint_angles(R3_6)
    
    return (theta_4, theta_5, theta_6)
    

