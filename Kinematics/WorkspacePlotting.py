import numpy as np
import matplotlib.pyplot as plt

import DirectForwardKinematics as FK
import TrajectoryTest as TT
import RobotConstants as RC

steps = 5

shoulder_angles = np.linspace(RC.shoulder_joint_limits[0],RC.shoulder_joint_limits[1],steps)
upper_angles = np.linspace(RC.upper_joint_limits[0],RC.upper_joint_limits[1],steps)
lower_angles = np.linspace(RC.lower_joint_limits[0],RC.lower_joint_limits[1],steps)
wrist_angles = np.linspace(RC.wrist_joint_limits[0],RC.wrist_joint_limits[1],steps)

points = []
theta_gripper = 0

for theta_shoulder in shoulder_angles:
    for theta_upper in upper_angles:
        for theta_lower in lower_angles:
            for theta_wrist in wrist_angles:
                Transformation = FK.ForwardKinematics(np.array([theta_shoulder, theta_upper, theta_lower, theta_wrist, theta_gripper]))
                Point = np.array([Transformation[0,3],Transformation[1,3],Transformation[2,3]])
                points.append(Point)

points = np.array(points)
print(points.shape)

fig = plt.figure()

cartesian_shape = TT.cartesian_trajectory

ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])
ax.scatter(cartesian_shape[:,0], cartesian_shape[:,1], cartesian_shape[:,2])

ax.axis('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()