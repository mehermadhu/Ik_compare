
import numpy as np
from scipy.spatial.transform import Rotation as R
def azimuth_angle_calculation(wrist_centre_positions): 
    azimuth_angles = np.arctan2(wrist_centre_positions[:, 1], wrist_centre_positions[:, 0])
    return azimuth_angles

def radial_distance_calculation(wrist_centre_positions): 
    radial_distances = np.sqrt((wrist_centre_positions[:, 1])**2 + (wrist_centre_positions[:, 0])**2)
    return radial_distances

def elevation_angle_calculation(wrist_centre_positions):
    sqrt_part = np.sqrt(wrist_centre_positions[:,0]**2 + wrist_centre_positions[:,1]**2)
    offset = 1e-8
    elevation_angles = np.arctan2(wrist_centre_positions[:,2], sqrt_part + offset)
    return elevation_angles
def rotation_matrix(roll, pitch, yaw):
    """ Returns a rotation matrix that encodes the roll, pitch, yaw orientation """
  
    R_x = np.array([[1,                  0,                 0                  ],
                    [0,         np.cos(roll), np.sin(roll)],
                    [0,        -np.sin(roll), np.cos(roll)]
                    ])
              
    R_y = np.array([[np.cos(pitch),    0, -np.sin(pitch)],
            [0,                     1,      0                  ],
            [np.sin(pitch),   0, np.cos(pitch)]
            ])
         
    R_z = np.array([[np.cos(yaw), np.sin(yaw),    0],
            [-np.sin(yaw), np.cos(yaw),    0],
            [0,                     0,                      1]
            ])
            
    R_o = np.dot(R_z, np.dot( R_y, R_x ))
    
    return R_o
def calculate_global_position(P, O, offset):
    """ Calculate the global position of the end of the link """
  

  
    # Build rotation matrix from orientation
    R_o = rotation_matrix(*O)
  
    # Define the position of the other end of the link in the local frame at P
    Q_local = np.array(offset)
  
    # Calculate the position of the other end of the link in the global reference frame
    Q_global = np.dot(R_o, Q_local) + P
  
    return Q_global

def rotation_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1]
    ])

def wrist_center_position(target_position, target_orientation):
    R_matrix = R.from_euler('zyx', target_orientation,degrees=False).as_matrix()
    a4 = 1
    displacement = np.array([a4, 0, 0])

    # Transform the displacement to the end effector coordinate frame
    transformation = np.dot(R_matrix, displacement)

    # Subtract the transformation from the end effector's position to get wrist center 
    wrist_center = target_position - transformation

    return wrist_center