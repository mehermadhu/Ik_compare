import unittest
from robot_model import move_around

class TestMoveAround(unittest.TestCase):

    def test_normal_case(self):
        # Define your _a, _d, _alpha variables
        # Expected _azimuth_joint_angles, _azimuth_wrist_centre_positions 
        _azimuth_joint_angles,_azimuth_wrist_centre_positions = move_around(_a, _d, _alpha)
        # Check the returned value is as expected
        self.assertEqual(_azimuth_joint_angles, expected_azimuth_joint_angles)
        self.assertEqual(_azimuth_wrist_centre_positions, expected_azimuth_wrist_centre_positions)

if __name__ == '__main__':
    unittest.main()