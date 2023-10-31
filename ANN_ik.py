import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from robot_model import Robot
import itertools
# Constants
manipulator = Robot()
# irb 120
manipulator.a = [0,4,4, 0, 0, 1] 
manipulator.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2,0] 
manipulator.d = [0, 0, 0, 0,0,0]
manipulator.joint_offsets = [0,0,0,0,0,0]

N = 10 # number of samples for each joint angle

# Step 1 - Generate Training Data
theta_range = np.linspace(-np.pi, np.pi, N)
inputs = []
outputs = []

for theta in itertools.product(theta_range, repeat=6):
    pos = manipulator.forward_kinematics(np.array(theta), joint_indx=5)
    inputs.append(np.array([pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]))
    outputs.append(np.array(theta))

inputs = np.array(inputs)
outputs = np.array(outputs)

# Step 2 - Define and Train ANN model
model = Sequential([Dense(32, input_dim=6, activation='relu'), Dense(6, activation='linear')])

model.compile(loss='mse', optimizer='adam')
model.fit(inputs, outputs, epochs=200, batch_size=32, validation_split=0.2)

# Step 3 - Test ANN model
test_points_cartesian = np.random.uniform(-np.pi, np.pi, (10, 6)) # 10 random 6-DoF instances between -pi and pi
ann_pred_angles = model.predict(test_points_cartesian)
# Replace these with correct formulas for forward kinematics of 6-DoF manipulator
#test_points_cartesian = np.zeros_like(test_thetas)
#for i, theta in enumerate(test_thetas):
#    result = manipulator.forward_kinematics(theta,5)
#    test_points_cartesian[i] = np.array([result[0], result[1], result[2], result[3], result[4], result[5]])
#ann_pred_angles = model.predict(test_points_cartesian)