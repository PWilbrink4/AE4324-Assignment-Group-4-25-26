import numpy as np
import matplotlib.pyplot as plt
import InverseKinematics as IK

number_waypoints = 10
radius_circle = 0.1
x_offset = 0
z_offset = 0.25
y_offset = 0.2

pitch = 0
roll = 0
yaw=0

cartesian_trajectory = np.zeros((number_waypoints+1, 5))
jointspace_trajectory_1 = np.zeros((number_waypoints, 5))
jointspace_trajectory_2 = np.zeros((number_waypoints, 5))
jointspace_trajectory_3 = np.zeros((number_waypoints, 5))
jointspace_trajectory_4 = np.zeros((number_waypoints, 5))

for i in range(number_waypoints):
    cartesian_trajectory[i, 0] = x_offset+radius_circle*np.cos(2*np.pi*i/(number_waypoints-1))
    cartesian_trajectory[i, 1] = y_offset+radius_circle*np.sin(2*np.pi*i/(number_waypoints-1))
    cartesian_trajectory[i, 2] = z_offset-0.1*np.sin(2*np.pi*i/(number_waypoints-1))
    cartesian_trajectory[i, 3] = pitch
    cartesian_trajectory[i, 4] = roll

if __name__ == "__main__":
    for i in range(number_waypoints):
        jointspace_trajectory_1[i], jointspace_trajectory_2[i], jointspace_trajectory_3[i], jointspace_trajectory_4[i] = IK.InverseKinematics(cartesian_trajectory[i])

    print(jointspace_trajectory_1)

    print(jointspace_trajectory_2)

'''WORKING TRAJECTORY 1 - Circle in x,y'''
# number_waypoints = 10
# radius_circle = 0.1
# z_offset = 0.3
# y_offset = 0.2
#
# pitch = 0
# roll = 0
# yaw=0
#
# cartesian_trajectory = np.zeros((number_waypoints, 5))
# jointspace_trajectory_1 = np.zeros((number_waypoints, 5))
# jointspace_trajectory_2 = np.zeros((number_waypoints, 5))
# jointspace_trajectory_3 = np.zeros((number_waypoints, 5))
# jointspace_trajectory_4 = np.zeros((number_waypoints, 5))
#
# for i in range(number_waypoints):
#     cartesian_trajectory[i, 0] = radius_circle*np.cos(2*np.pi*i/number_waypoints)
#     cartesian_trajectory[i, 1] = radius_circle*np.sin(2*np.pi*i/number_waypoints)+y_offset
#     cartesian_trajectory[i, 2] = z_offset
#     cartesian_trajectory[i, 3] = pitch
#     cartesian_trajectory[i, 4] = roll

'''WORKING TRAJECTORY 2 - Circle in x,y with z variation'''
# number_waypoints = 10
# radius_circle = 0.1
# x_offset = 0
# z_offset = 0.25
# y_offset = 0.2
#
# pitch = 0
# roll = 0
# yaw=0
#
# cartesian_trajectory = np.zeros((number_waypoints, 5))
# jointspace_trajectory_1 = np.zeros((number_waypoints, 5))
# jointspace_trajectory_2 = np.zeros((number_waypoints, 5))
# jointspace_trajectory_3 = np.zeros((number_waypoints, 5))
# jointspace_trajectory_4 = np.zeros((number_waypoints, 5))
#
# for i in range(number_waypoints):
#     cartesian_trajectory[i, 0] = x_offset+radius_circle*np.cos(2*np.pi*i/number_waypoints)
#     cartesian_trajectory[i, 1] = y_offset+radius_circle*np.sin(2*np.pi*i/number_waypoints)
#     cartesian_trajectory[i, 2] = z_offset-0.1*np.sin(2*np.pi*i/number_waypoints)
#     cartesian_trajectory[i, 3] = pitch
#     cartesian_trajectory[i, 4] = roll
