import numpy as np
import matplotlib.pyplot as plt

import DirectForwardKinematics as FK
import TrajectoryTest as TT
import RobotConstants as RC


'''This file is used to generate plots of the workspace based on the forward kinematics and joint constraints
'''

steps = 20
# shoulder_angles = np.linspace(RC.shoulder_joint_limits[0],RC.shoulder_joint_limits[1],2*steps)
# upper_angles = np.linspace(RC.upper_joint_limits[0],RC.upper_joint_limits[1],steps)
# lower_angles = np.linspace(RC.lower_joint_limits[0],RC.lower_joint_limits[1],steps)
# wrist_angles = np.linspace(RC.wrist_joint_limits[0],RC.wrist_joint_limits[1],steps)
shoulder_angles = np.linspace(-np.pi,np.pi,2*steps)
upper_angles = np.linspace(-np.pi,np.pi,steps)
lower_angles = np.linspace(-np.pi,np.pi,steps)
wrist_angles = np.linspace(-np.pi,np.pi,steps)

points = []
yz_points = []
xz_points = []
xy_points = []

theta_gripper = 0
threshold = 0.01

for theta_shoulder in shoulder_angles:
    print(theta_shoulder)
    for theta_upper in upper_angles:
        for theta_lower in lower_angles:
            for theta_wrist in wrist_angles:
                Transformation = FK.ForwardKinematics(np.array([theta_shoulder, theta_upper, theta_lower, theta_wrist, theta_gripper]))
                Point = np.array([Transformation[0,3],Transformation[1,3],Transformation[2,3]])
                points.append(Point)
                if abs(Point[0]) < threshold:
                    yz_points.append(Point)
                if abs(Point[1]) < threshold:
                    xz_points.append(Point)
                if abs(Point[2]) < threshold:
                    xy_points.append(Point)

print("plotting")

points = np.array(points)
yz_points = np.array(yz_points)
xz_points = np.array(xz_points)
xy_points = np.array(xy_points)

if not steps > 10:
    fig = plt.figure()

    cartesian_shape = TT.cartesian_trajectory

    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    # ax.scatter(cartesian_shape[:,0], cartesian_shape[:,1], cartesian_shape[:,2])
    ax.scatter(1,0,0,color="red",label="Global X-Axis")
    ax.scatter(0,1,0,color="green",label="Global Y-Axis")
    ax.scatter(0,0,1,color="blue",label="Global Z-Axis")

    ax.axis('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    fig.tight_layout()

fig1,ax1 = plt.subplots(1,3,figsize=(18,6),sharex=True)

ax1[0].set_title("Workspace on the XY plane")
ax1[0].scatter(points[:,0], points[:,1],label="Projection")
# ax1[0].scatter(xy_points[:,0], xy_points[:,1],label="On XY-plane")
ax1[0].scatter(0.05,0,label="Global X-Axis",color="red")
ax1[0].scatter(0,0.05,label="Global Y-Axis",color="green")
ax1[0].scatter(0,0,label="Global Z-Axis",color="blue")
ax1[0].set_xlim(-0.75,0.75)
ax1[0].set_ylim(-0.75,0.75)
ax1[0].set_xlabel('X')
ax1[0].set_ylabel('Y')
ax1[0].axis('equal')
ax1[0].legend()

ax1[1].set_title("Workspace on the XZ plane")
ax1[1].scatter(points[:,0], points[:,2],label="Projection")
# ax1[1].scatter(xz_points[:,0], xz_points[:,2],label="On XZ-plane")
ax1[1].scatter(0.05,0,label="Global X-Axis",color="red")
ax1[1].scatter(0,0,label="Global Y-Axis",color="green")
ax1[1].scatter(0,0.05,label="Global Z-Axis",color="blue")
ax1[1].set_xlim(-0.75,0.75)
ax1[1].set_ylim(-0.75,0.75)
ax1[1].set_xlabel('X')
ax1[1].set_ylabel('Z')
ax1[1].axis('equal')
ax1[1].legend()

ax1[2].set_title("Workspace on the YZ plane")
ax1[2].scatter(points[:,1], points[:,2],label="Projection")
# ax1[2].scatter(yz_points[:,1], yz_points[:,2],label="On YZ-plane")
ax1[2].scatter(0,0,label="Global X-Axis",color="red")
ax1[2].scatter(0.05,0,label="Global Y-Axis",color="green")
ax1[2].scatter(0,0.05,label="Global Z-Axis",color="blue")
ax1[2].set_xlim(-0.75,0.75)
ax1[2].set_ylim(-0.75,0.75)
ax1[2].set_xlabel('Y')
ax1[2].set_ylabel('Z')
ax1[2].axis('equal')
ax1[2].legend()

fig1.tight_layout()

plt.show()