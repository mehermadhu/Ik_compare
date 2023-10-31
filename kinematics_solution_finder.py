import numpy as np

def find_solution(poly, value, joint_limits):
    poly_funs = np.poly1d(poly)
    solution = poly_funs(value)
    print(solution)
    joint_min, joint_max = joint_limits
#    if solution < joint_min:
#        solution = joint_min
#    if solution > joint_max:
#        solution = joint_max
    return solution

def find_best_solution(poly, value,  curr_joint_angle,joint_limits):
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
        



def my_IK(robot,target_position, target_orientation, curr_joint_angles,ax):
    # First 3 joints
    azimuth = np.arctan2(target_position[1], target_position[0])  # Get azimuth
    elevation = np.arcsin(target_position[2] / np.linalg.norm(target_position))  # Get elevation
    radius = np.linalg.norm(target_position)  # Calculate radius

#    theta1 = find_best_solution(robot.azimuth_coeffs, azimuth, curr_joint_angles[0], robot.joint_limits[0])
#    theta2 = find_best_solution(robot.elevation_coeffs, elevation, curr_joint_angles[1], robot.joint_limits[1])
#    del_theta2 = find_best_solution(robot.rad_coeffs_joint1, radius, curr_joint_angles[1], robot.joint_limits[1])
#    theta3 = find_best_solution(robot.rad_coeffs_joint2, radius, curr_joint_angles[2], robot.joint_limits[2])

    theta1 = find_solution(robot.azimuth_coeffs, azimuth, robot.joint_limits[0])
    theta2 = find_solution(robot.elevation_coeffs, elevation, robot.joint_limits[1])
    del_theta2 = find_solution(robot.rad_coeffs_joint1, radius, robot.joint_limits[1])
    theta3 = find_solution(robot.rad_coeffs_joint2, radius, robot.joint_limits[2])

    if None in [theta1, theta2,del_theta2, theta3]:  # If any joint parameter is None, return None
        return None
    else:
    	theta2 = theta2+del_theta2
    
    # Extract roll, pitch, yaw from target orientation
    roll, pitch, yaw = target_orientation #np.array([np.pi/2, 0, 0])  # Example values
    #Last 3 joints
    #theta4, theta5, theta6 = robot.calculate_last_three_joints(target_position,theta1, theta2, theta3, roll, pitch, yaw,ax)
    theta4, theta5, theta6 = robot.calculate_last_three_joints(theta1, theta2, theta3, roll, pitch, yaw)

    return np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    