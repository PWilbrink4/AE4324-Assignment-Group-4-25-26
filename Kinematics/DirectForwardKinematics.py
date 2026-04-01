import matplotlib.pyplot as plt
import numpy as np

'''This file defines the functions for the forward kinematics, rotation matrices, and the code for the python visualisations seen throughout the report
    Running will plot some example positions
'''

def RotationMatrix_X(angle):
    Rx = np.matrix([
        [1,0,0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    return Rx

def RotationMatrix_Y(angle):
    Ry = np.matrix([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    return Ry

def RotationMatrix_Z(angle):
    Rz = np.matrix([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return Rz

def TranslationMatrix(x, y, z):
    P = np.matrix([
        [x],
        [y],
        [z]
    ])
    return P

def HomogeneousTransformation(rotation_matrix, translation_matrix):
    T = rotation_matrix.copy()
    T = np.hstack((T, translation_matrix))
    T = np.vstack((T,np.matrix([[0,0,0,1]])))
    return T

def ForwardKinematics(q_vector):
    #R_i_j means rotation of frame j with respect to frame i
    theta_shoulder = q_vector[0]
    theta_upper = q_vector[1]
    theta_lower = q_vector[2]
    theta_wrist = q_vector[3]
    theta_gripper = q_vector[4]

    Zero_translation = TranslationMatrix(0, 0, 0)

    #R_i_j means rotation of frame j with respect to frame i
    R_world_base = RotationMatrix_Z(np.pi)
    P_world_base = TranslationMatrix(0, 0, 0)
    T_world_base = HomogeneousTransformation(R_world_base, P_world_base)

    R_base_shoulder = RotationMatrix_X(0)
    P_base_shoulder = TranslationMatrix(0, -0.0452, 0.0165)
    T_base_shoulder = HomogeneousTransformation(R_base_shoulder, P_base_shoulder)

    R_shoulder_upper = RotationMatrix_Y(-np.pi / 2)
    P_shoulder_upper = TranslationMatrix(0, -0.0306, 0.1025)
    T_shoulder_upper = HomogeneousTransformation(RotationMatrix_Z(theta_shoulder),Zero_translation)*HomogeneousTransformation(R_shoulder_upper, P_shoulder_upper)

    R_upper_lower = RotationMatrix_Z(0)
    P_upper_lower = TranslationMatrix(0.11257, -0.028, 0)
    T_upper_lower = HomogeneousTransformation(RotationMatrix_Z(theta_upper), Zero_translation)*HomogeneousTransformation(R_upper_lower, P_upper_lower)

    R_lower_wrist = RotationMatrix_Z(np.pi / 2)
    P_lower_wrist = TranslationMatrix(0.0052, -0.1349, 0)
    T_lower_wrist = HomogeneousTransformation(RotationMatrix_Z(theta_lower),Zero_translation)*HomogeneousTransformation(R_lower_wrist, P_lower_wrist)

    R_wrist_gripper = RotationMatrix_Y(-np.pi / 2)
    P_wrist_gripper = TranslationMatrix(-0.0601, 0, 0)
    T_wrist_gripper = HomogeneousTransformation(RotationMatrix_Z(theta_wrist),Zero_translation)*HomogeneousTransformation(R_wrist_gripper, P_wrist_gripper)

    R_gripper_grippercenter = RotationMatrix_Z(0)
    P_gripper_grippercenter = TranslationMatrix(0, 0, 0.075)
    T_gripper_grippercenter = HomogeneousTransformation(RotationMatrix_Z(theta_gripper),Zero_translation)*HomogeneousTransformation(R_gripper_grippercenter, P_gripper_grippercenter)

    T_world_grippercenter = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist*T_wrist_gripper*T_gripper_grippercenter

    return T_world_grippercenter


def ForwardKinematicsFullOutput_PreRotationFrame(q_vector):
    theta_shoulder = q_vector[0]
    theta_upper = q_vector[1]
    theta_lower = q_vector[2]
    theta_wrist = q_vector[3]
    theta_gripper = q_vector[4]

    Zero_translation = TranslationMatrix(0, 0, 0)

    #R_i_j means rotation of frame j with respect to frame i
    R_world_base = RotationMatrix_Z(np.pi)
    P_world_base = TranslationMatrix(0, 0, 0)
    T_world_base = HomogeneousTransformation(R_world_base, P_world_base)

    R_base_shoulder = RotationMatrix_X(0)
    P_base_shoulder = TranslationMatrix(0, -0.0452, 0.0165)
    T_base_shoulder = HomogeneousTransformation(R_base_shoulder, P_base_shoulder)

    R_shoulder_upper = RotationMatrix_Y(-np.pi / 2)
    P_shoulder_upper = TranslationMatrix(0, -0.0306, 0.1025)
    T_shoulder_upper = HomogeneousTransformation(RotationMatrix_Z(theta_shoulder),Zero_translation)*HomogeneousTransformation(R_shoulder_upper, P_shoulder_upper)

    R_upper_lower = RotationMatrix_Z(0)
    P_upper_lower = TranslationMatrix(0.11257, -0.028, 0)
    T_upper_lower = HomogeneousTransformation(RotationMatrix_Z(theta_upper), Zero_translation)*HomogeneousTransformation(R_upper_lower, P_upper_lower)

    R_lower_wrist = RotationMatrix_Z(np.pi / 2)
    P_lower_wrist = TranslationMatrix(0.0052, -0.1349, 0)
    T_lower_wrist = HomogeneousTransformation(RotationMatrix_Z(theta_lower),Zero_translation)*HomogeneousTransformation(R_lower_wrist, P_lower_wrist)

    R_wrist_gripper = RotationMatrix_Y(-np.pi / 2)
    P_wrist_gripper = TranslationMatrix(-0.0601, 0, 0)
    T_wrist_gripper = HomogeneousTransformation(RotationMatrix_Z(theta_wrist),Zero_translation)*HomogeneousTransformation(R_wrist_gripper, P_wrist_gripper)

    R_gripper_grippercenter = RotationMatrix_Z(0)
    P_gripper_grippercenter = TranslationMatrix(0, 0, 0.075)
    T_gripper_grippercenter = HomogeneousTransformation(RotationMatrix_Z(theta_gripper),Zero_translation)*HomogeneousTransformation(R_gripper_grippercenter, P_gripper_grippercenter)

    T_world_shoulder = T_world_base*T_base_shoulder
    T_world_upper = T_world_base*T_base_shoulder*T_shoulder_upper
    T_world_lower = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower
    T_world_wrist = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist
    T_world_gripper = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist*T_wrist_gripper
    T_world_grippercenter = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist*T_wrist_gripper*T_gripper_grippercenter

    return T_world_base, T_world_shoulder, T_world_upper, T_world_lower, T_world_wrist, T_world_gripper, T_world_grippercenter

def ForwardKinematicsFullOutput_PostRotationFrame(q_vector):
    theta_shoulder = q_vector[0]
    theta_upper = q_vector[1]
    theta_lower = q_vector[2]
    theta_wrist = q_vector[3]
    theta_gripper = q_vector[4]

    Zero_translation = TranslationMatrix(0, 0, 0)

    #R_i_j means rotation of frame j with respect to frame i
    R_world_base = RotationMatrix_Z(np.pi)
    P_world_base = TranslationMatrix(0, 0, 0)
    T_world_base = HomogeneousTransformation(R_world_base, P_world_base)

    R_base_shoulder = RotationMatrix_X(0) * RotationMatrix_Z(theta_shoulder)
    P_base_shoulder = TranslationMatrix(0, -0.0452, 0.0165)
    T_base_shoulder = HomogeneousTransformation(R_base_shoulder, P_base_shoulder)

    R_shoulder_upper = RotationMatrix_Y(-np.pi / 2) * RotationMatrix_Z(theta_upper)
    P_shoulder_upper = TranslationMatrix(0, -0.0306, 0.1025)
    T_shoulder_upper = HomogeneousTransformation(R_shoulder_upper, P_shoulder_upper)

    R_upper_lower = RotationMatrix_Z(theta_lower)
    P_upper_lower = TranslationMatrix(0.11257, -0.028, 0)
    T_upper_lower = HomogeneousTransformation(R_upper_lower, P_upper_lower)

    R_lower_wrist = RotationMatrix_Z(np.pi / 2) * RotationMatrix_Z(theta_wrist)
    P_lower_wrist = TranslationMatrix(0.0052, -0.1349, 0)
    T_lower_wrist = HomogeneousTransformation(R_lower_wrist, P_lower_wrist)

    R_wrist_gripper = RotationMatrix_Y(-np.pi / 2) * RotationMatrix_Z(theta_gripper)
    P_wrist_gripper = TranslationMatrix(-0.0601, 0, 0)
    T_wrist_gripper = HomogeneousTransformation(R_wrist_gripper, P_wrist_gripper)

    R_gripper_grippercenter = RotationMatrix_Z(0)
    P_gripper_grippercenter = TranslationMatrix(0, 0, 0.075)
    T_gripper_grippercenter = HomogeneousTransformation(R_gripper_grippercenter, P_gripper_grippercenter)

    T_world_shoulder = T_world_base*T_base_shoulder
    T_world_upper = T_world_base*T_base_shoulder*T_shoulder_upper
    T_world_lower = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower
    T_world_wrist = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist
    T_world_gripper = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist*T_wrist_gripper
    T_world_grippercenter = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist*T_wrist_gripper*T_gripper_grippercenter

    return T_world_base, T_world_shoulder, T_world_upper, T_world_lower, T_world_wrist, T_world_gripper, T_world_grippercenter

def XYZ_from_T(transformation):
    point = transformation*np.array([[0],[0],[0],[1]])
    return point[0:3]

def refframe_from_T(transformation):
    x_vector = transformation*np.array([[0.03],[0],[0],[1]])
    y_vector = transformation*np.array([[0],[0.03],[0],[1]])
    z_vector = transformation*np.array([[0],[0],[0.03],[1]])
    return x_vector[0:3], y_vector[0:3], z_vector[0:3]

def plot_robot_position(q_vector):
    TJ0, TJ1, TJ2, TJ3, TJ4, TJ5, TJ6 = ForwardKinematicsFullOutput_PostRotationFrame(q_vector)

    x, y, z = XYZ_from_T(TJ6)[0,0], XYZ_from_T(TJ6)[1,0], XYZ_from_T(TJ6)[2,0]
    pitch = q_vector[1]+q_vector[2]+q_vector[3]
    roll = q_vector[4]


    state = np.array([x,y,z,pitch,roll])
    np.set_printoptions(suppress=True)
    print(f'x, y, z, vis pitch, vis roll: {state}')
    rot_x, rot_y, rot_z = Fixed_angles_from_matrix(TJ6)
    print(f'Rotation XYZ: {rot_x}, {rot_y}, {rot_z}')

    np.set_printoptions(precision=5)

    points = np.array([
            [[0],[0],[0]],
            XYZ_from_T(TJ0),
            XYZ_from_T(TJ1),
            XYZ_from_T(TJ2),
            XYZ_from_T(TJ3),
            XYZ_from_T(TJ4),
            XYZ_from_T(TJ5),
            XYZ_from_T(TJ6)])

    X_points = np.array([
        [[0.1],[0],[0]],
        refframe_from_T(TJ0)[0],
        refframe_from_T(TJ1)[0],
        refframe_from_T(TJ2)[0],
        refframe_from_T(TJ3)[0],
        refframe_from_T(TJ4)[0],
        refframe_from_T(TJ5)[0],
        refframe_from_T(TJ6)[0]
    ])

    Y_points = np.array([
        [[0],[0.1],[0]],
        refframe_from_T(TJ0)[1],
        refframe_from_T(TJ1)[1],
        refframe_from_T(TJ2)[1],
        refframe_from_T(TJ3)[1],
        refframe_from_T(TJ4)[1],
        refframe_from_T(TJ5)[1],
        refframe_from_T(TJ6)[1]
    ])

    Z_points = np.array([
        [[0],[0],[0.1]],
        refframe_from_T(TJ0)[2],
        refframe_from_T(TJ1)[2],
        refframe_from_T(TJ2)[2],
        refframe_from_T(TJ3)[2],
        refframe_from_T(TJ4)[2],
        refframe_from_T(TJ5)[2],
        refframe_from_T(TJ6)[2]
    ])

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[1:-1, 0], points[1:-1, 1], points[1:-1, 2],label="Joint",color="black")
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], label="EE", color="purple",marker='*')
    ax.plot(points[1:, 0], points[1:, 1], points[1:, 2],color="black")
    ax.scatter(X_points[1:, 0], X_points[1:, 1], X_points[1:, 2],label="Frame X-axis",color="red")
    ax.scatter(Y_points[1:, 0], Y_points[1:, 1], Y_points[1:, 2],label="Frame Y-axis",color="green")
    ax.scatter(Z_points[1:, 0], Z_points[1:, 1], Z_points[1:, 2],label="Frame Z-axis",color="blue")

    ax.scatter(points[0, 0], points[0, 1], points[0, 2],color="black",marker='d')
    ax.scatter(X_points[0, 0], X_points[0, 1], X_points[0, 2],color="red",marker='d')
    ax.scatter(Y_points[0, 0], Y_points[0, 1], Y_points[0, 2],color="green",marker='d')
    ax.scatter(Z_points[0, 0], Z_points[0, 1], Z_points[0, 2],color="blue",marker='d')
    plt.axis('equal')

    ax.set_title(f"EE: {state},\n q: {q_vector}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    #plt.show()
    np.set_printoptions(suppress=False)
    np.set_printoptions(precision=8)

def plot_robot_position_and_target(q_vector,target):
    TJ0, TJ1, TJ2, TJ3, TJ4, TJ5, TJ6 = ForwardKinematicsFullOutput_PostRotationFrame(q_vector)

    x, y, z = XYZ_from_T(TJ6)[0,0], XYZ_from_T(TJ6)[1,0], XYZ_from_T(TJ6)[2,0]
    pitch = q_vector[1]+q_vector[2]+q_vector[3]
    roll = q_vector[4]

    targetpoint = np.array(target[0:3])
    targetrotation = Fixed_angles_to_rotation_matrix(target[3],target[4],target[5])
    targetxaxis = targetpoint+np.dot(targetrotation, np.array([0.05, 0, 0]))
    targetyaxis = targetpoint+np.dot(targetrotation, np.array([0, 0.05, 0]))
    targetzaxis = targetpoint+np.dot(targetrotation, np.array([0, 0, 0.05]))

    state = np.array([x,y,z,pitch,roll])
    np.set_printoptions(suppress=True)
    print(f'x, y, z, vis pitch, vis roll: {state}')
    rot_x, rot_y, rot_z = Fixed_angles_from_matrix(TJ6)
    print(f'Rotation XYZ: {rot_x}, {rot_y}, {rot_z}')

    np.set_printoptions(precision=5)

    points = np.array([
            [[0],[0],[0]],
            XYZ_from_T(TJ0),
            XYZ_from_T(TJ1),
            XYZ_from_T(TJ2),
            XYZ_from_T(TJ3),
            XYZ_from_T(TJ4),
            XYZ_from_T(TJ5),
            XYZ_from_T(TJ6)])

    X_points = np.array([
        [[0.1],[0],[0]],
        refframe_from_T(TJ0)[0],
        refframe_from_T(TJ1)[0],
        refframe_from_T(TJ2)[0],
        refframe_from_T(TJ3)[0],
        refframe_from_T(TJ4)[0],
        refframe_from_T(TJ5)[0],
        refframe_from_T(TJ6)[0]
    ])

    Y_points = np.array([
        [[0],[0.1],[0]],
        refframe_from_T(TJ0)[1],
        refframe_from_T(TJ1)[1],
        refframe_from_T(TJ2)[1],
        refframe_from_T(TJ3)[1],
        refframe_from_T(TJ4)[1],
        refframe_from_T(TJ5)[1],
        refframe_from_T(TJ6)[1]
    ])

    Z_points = np.array([
        [[0],[0],[0.1]],
        refframe_from_T(TJ0)[2],
        refframe_from_T(TJ1)[2],
        refframe_from_T(TJ2)[2],
        refframe_from_T(TJ3)[2],
        refframe_from_T(TJ4)[2],
        refframe_from_T(TJ5)[2],
        refframe_from_T(TJ6)[2]
    ])

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.scatter(targetpoint[0], targetpoint[1], targetpoint[2], color="black", marker='x',label="Target")
    ax.scatter(targetxaxis[0,0], targetxaxis[0,1], targetxaxis[0,2], color="red", marker='x',label="Target X-axis")
    ax.scatter(targetyaxis[0,0], targetyaxis[0,1], targetyaxis[0,2], color="green", marker='x', label="Target Y-axis")
    ax.scatter(targetzaxis[0,0], targetzaxis[0,1], targetzaxis[0,2], color="blue", marker='x', label="Target Z-axis")

    ax.scatter(points[1:-1, 0], points[1:-1, 1], points[1:-1, 2],label="Joint",color="black")
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], label="EE", color="purple",marker='*')
    ax.plot(points[1:, 0], points[1:, 1], points[1:, 2],color="black")
    ax.scatter(X_points[1:, 0], X_points[1:, 1], X_points[1:, 2],label="Frame X-axis",color="red")
    ax.scatter(Y_points[1:, 0], Y_points[1:, 1], Y_points[1:, 2],label="Frame Y-axis",color="green")
    ax.scatter(Z_points[1:, 0], Z_points[1:, 1], Z_points[1:, 2],label="Frame Z-axis",color="blue")

    ax.scatter(points[0, 0], points[0, 1], points[0, 2],color="black",marker='d')
    ax.scatter(X_points[0, 0], X_points[0, 1], X_points[0, 2],color="red",marker='d')
    ax.scatter(Y_points[0, 0], Y_points[0, 1], Y_points[0, 2],color="green",marker='d')
    ax.scatter(Z_points[0, 0], Z_points[0, 1], Z_points[0, 2],color="blue",marker='d')

    plt.axis('equal')

    ax.set_title(f"EE: {state},\n q: {q_vector}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    #plt.show()
    np.set_printoptions(suppress=False)
    np.set_printoptions(precision=8)

def Fixed_angles_to_rotation_matrix(rot_x, rot_y, rot_z):
    ca = np.cos(rot_x)
    sa = np.sin(rot_x)
    cb = np.cos(rot_y)
    sb = np.sin(rot_y)
    cg = np.cos(rot_z)
    sg = np.sin(rot_z)

    R = np.matrix([
        [cb*cg,-cb*sg,sb],
        [sa*sb*cg+ca*sg,-sa*sb*sg+ca*cg,-sa*cb],
        [-ca*sb*cg+sa*sg,ca*sb*sg+sa*cg,ca*cb]
    ])
    return R

def Relative_angles_to_rotation_matrix(yaw, pitch, roll):
    ca = np.cos(yaw)
    sa = np.sin(yaw)
    cb = np.cos(pitch)
    sb = np.sin(pitch)
    cg = np.cos(roll)
    sg = np.sin(roll)

    R = np.matrix([
        [ca*cb,ca*sb*sg-sa*cg,ca*sb*cg+sa*sg],
        [sa*cb,sa*sb*sg+ca*cg,sa*sb*cg-ca*sg],
        [-sb,cb*sg,cb*cg]
    ])
    return R

def Relative_angles_from_matrix(R):
    yaw = np.arctan2(R[1,0],R[0,0])
    pitch = -np.arcsin(R[2,0])
    roll = np.arctan2(R[2,1],R[2,2])
    return yaw, pitch, roll

def Fixed_angles_from_matrix(R):
    rot_x = -np.arctan2(R[1,2],R[2,2])
    rot_y = np.arcsin(R[0,2])
    rot_z = -np.arctan2(R[0,1],R[0,0])
    return rot_x, rot_y, rot_z

# def SixStates_to_FiveStates(state):
#     x = state[0]
#     y = state[1]
#     z = state[2]
#     rot_x, rot_y, rot_z = state[3:6]
#     z_unit_vector = [0,0,1]
#     y_unit_vector = [0,1,0]
#     R = Fixed_angles_to_rotation_matrix(rot_x, rot_y, rot_z)
#     EE_z_unit_vector = np.dot(R, z_unit_vector).flatten()
#     EE_y_unit_vector = np.dot(R, y_unit_vector).flatten()
#     pitch = np.arcsin(np.abs(np.dot(EE_z_unit_vector, z_unit_vector))/(np.linalg.norm(z_unit_vector)*np.linalg.norm(EE_z_unit_vector)))
#     projection_pitch_xy = EE_z_unit_vector-((np.dot(EE_z_unit_vector,z_unit_vector))/(np.linalg.norm(z_unit_vector)**2))*z_unit_vector
#     EE_plane_yz_normal = np.cross(EE_z_unit_vector, projection_pitch_xy).flatten()
#     EE_y_initial_vector = np.cross(EE_z_unit_vector, EE_plane_yz_normal).flatten()
#     roll = np.arccos(np.dot(EE_y_unit_vector,EE_y_initial_vector)/(np.linalg.norm(EE_y_initial_vector)*np.linalg.norm(EE_y_unit_vector)))
#
#     yaw = np.arccos(np.dot(projection_pitch_xy,y_unit_vector)/(np.linalg.norm(projection_pitch_xy)*np.linalg.norm(y_unit_vector)))
#     # yaw, pitch, roll = Relative_angles_from_matrix(Fixed_angles_to_rotation_matrix(rot_x, rot_y, rot_z)*RotationMatrix_X(np.pi/2)) #Z, Y, X
#     # rel_pitch = roll #Around X
#     # rel_roll = -yaw #Around Z #TODO fix
#
#     rel_pitch = pitch[0,0]
#     rel_roll = roll[0,0]
#     geometry_rot_z = np.arctan2(y-0.0452,x)
#     print(f'yaw goal: {yaw}')
#     print(f'geometric yaw: {geometry_rot_z}')
#     print(f"yaw difference: {geometry_rot_z-yaw}")
#     new_state = np.array([x,y,z,rel_pitch,rel_roll])
#     print(new_state)
#     return new_state


if __name__ == '__main__':
    q_vector = np.radians(np.array([0,0,0,0,0])) #shoulder, upper, lower, wrist, gripper

    plot_robot_position(np.radians(np.array([0,0,0,0,0])))

    plot_robot_position(np.radians(np.array([45,45,-45,45,45])))

    plot_robot_position([0.0, 1.8326, -1.2217, -0.6109, -1.5707])

    plt.show()

    print(Fixed_angles_to_rotation_matrix(np.radians(30), np.radians(40), np.radians(-10)))
    print(np.degrees(Fixed_angles_from_matrix(Fixed_angles_to_rotation_matrix(np.radians(30), np.radians(40), np.radians(-10)))))
