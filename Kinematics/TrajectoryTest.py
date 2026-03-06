import numpy as np
import matplotlib.pyplot as plt
import InverseKinematics as IK

number_waypoints = 10
radius_circle = 0.2
y = 0.25
pitch = 0
roll = 0
z_offset = 0.15

cartesian_trajectory = np.zeros((number_waypoints, 5))
jointspace_trajectory_1 = np.zeros((number_waypoints, 5))
jointspace_trajectory_2 = np.zeros((number_waypoints, 5))

for i in range(number_waypoints):
    cartesian_trajectory[i, 0] = radius_circle*np.cos(2*np.pi*i/number_waypoints)
    cartesian_trajectory[i, 1] = y
    cartesian_trajectory[i, 2] = radius_circle*np.sin(2*np.pi*i/number_waypoints)+z_offset
    cartesian_trajectory[i, 3] = pitch
    cartesian_trajectory[i, 4] = roll

for i in range(number_waypoints):
    jointspace_trajectory_1[i], jointspace_trajectory_2[i] = IK.InverseKinematics(cartesian_trajectory[i])

print(jointspace_trajectory_1)

print(jointspace_trajectory_2)