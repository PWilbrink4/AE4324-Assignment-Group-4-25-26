import matplotlib.pyplot as plt
import numpy as np

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
    TJ0, TJ1, TJ2, TJ3, TJ4, TJ5, TJ6 = ForwardKinematicsFullOutput_PreRotationFrame(q_vector)

    x, y, z = XYZ_from_T(TJ6)[0,0], XYZ_from_T(TJ6)[1,0], XYZ_from_T(TJ6)[2,0]
    pitch = q_vector[1]+q_vector[2]+q_vector[3]
    roll = q_vector[4]


    state = np.array([x,y,z,pitch,roll])
    np.set_printoptions(suppress=True)
    print(f'EE state: {state}')
    np.set_printoptions(precision=5)

    points = np.array([
            XYZ_from_T(TJ0),
            XYZ_from_T(TJ1),
            XYZ_from_T(TJ2),
            XYZ_from_T(TJ3),
            XYZ_from_T(TJ4),
            XYZ_from_T(TJ5),
            XYZ_from_T(TJ6)])

    X_points = np.array([
        refframe_from_T(TJ0)[0],
        refframe_from_T(TJ1)[0],
        refframe_from_T(TJ2)[0],
        refframe_from_T(TJ3)[0],
        refframe_from_T(TJ4)[0],
        refframe_from_T(TJ5)[0],
        refframe_from_T(TJ6)[0]
    ])

    Y_points = np.array([
        refframe_from_T(TJ0)[1],
        refframe_from_T(TJ1)[1],
        refframe_from_T(TJ2)[1],
        refframe_from_T(TJ3)[1],
        refframe_from_T(TJ4)[1],
        refframe_from_T(TJ5)[1],
        refframe_from_T(TJ6)[1]
    ])

    Z_points = np.array([
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
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],label="Joint",color="black")
    ax.plot(points[:, 0], points[:, 1], points[:, 2],color="black")
    ax.scatter(X_points[:, 0], X_points[:, 1], X_points[:, 2],label="Frame X-axis",color="red")
    ax.scatter(Y_points[:, 0], Y_points[:, 1], Y_points[:, 2],label="Frame Y-axis",color="green")
    ax.scatter(Z_points[:, 0], Z_points[:, 1], Z_points[:, 2],label="Frame Z-axis",color="blue")
    plt.axis('equal')

    ax.set_title(f"EE: {state},\n q: {q_vector}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    #plt.show()
    np.set_printoptions(suppress=False)
    np.set_printoptions(precision=8)


if __name__ == '__main__':
    q_vector = np.radians(np.array([0,0,0,0,0])) #shoulder, upper, lower, wrist, gripper

    plot_robot_position(np.radians(np.array([0,0,0,0,0])))

    plot_robot_position(np.radians(np.array([45,45,-45,45,45])))

    plt.show()

