import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Constants
a1 = a2 = a3 = a4 = a5 = a6 = 1 # Equal link lengths

N = 100 # number of samples for each joint angle

# Step 1 - Generate Training Data
theta_range = np.linspace(-np.pi, np.pi, N)
theta_values = np.meshgrid(*[theta_range]*6) # Changed for 6 DoF
theta_values = [t.flatten() for t in theta_values]

# Replace these with correct formulas for forward kinematics of 6-DoF manipulator
x_values = np.sum([a*np.cos(theta) for a, theta in zip([a1, a2, a3, a4, a5, a6], theta_values)], axis=0)
y_values = np.sum([a*np.sin(theta) for a, theta in zip([a1, a2, a3, a4, a5, a6], theta_values)], axis=0)
z_values = np.ones_like(x_values)
rx_values = np.zeros_like(x_values)
ry_values = np.zeros_like(x_values)
rz_values = np.zeros_like(x_values)

inputs = np.array([x_values, y_values, z_values, rx_values, ry_values, rz_values]).T
outputs = np.array(theta_values).T

# Step 2 - Define and Train ANN model
model = Sequential([Dense(32, input_dim=6, activation='relu'), Dense(6, activation='linear')])

model.compile(loss='mse', optimizer='adam')

model.fit(inputs, outputs, epochs=200, batch_size=32, validation_split=0.2)

# Step 3 - Test ANN model
test_thetas = np.random.uniform(-np.pi, np.pi, (10, 6)) # 10 random values for each joint between -pi and pi
# Replace these with correct formulas for forward kinematics of 6-DoF manipulator
test_points_cartesian = np.zeros_like(test_thetas)
for i in range(6):
    test_points_cartesian[:,i] = (a1 * np.cos(test_thetas[:,i]) + a2 * np.cos(np.sum(test_thetas[:,:i+1], axis=1)))

ann_pred_angles = model.predict(test_points_cartesian)