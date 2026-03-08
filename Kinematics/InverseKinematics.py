import numpy as np
import matplotlib.pyplot as plt
from DirectForwardKinematics import *
import RobotConstants as RC
import TrajectoryTest as TT
from matplotlib.animation import FuncAnimation

# TODO: Make the selector for the chosen configuration
#       - Base the selection on distance to "old" configuration (Also feasibility ofc)
# TODO: Code it into one single sequence file

def InverseKinematics(EE_state_vector):
    x,y,z,pitch,roll = EE_state_vector

    #TODO discuss the pitch roll changing into xrot, yrot, zrot (me no like, doesnt make sense)
    #TODO Maybe check for pitch2 = pi-pitch ?

    # Robot joint positions
    y_shoulder = 0.0452

    # Since this is the only "yaw" joint rotation
    theta_shoulder_1 = np.arctan2(-x,y-y_shoulder) #shoulder joint
    theta_shoulder_2 = np.arctan2(x,-(y-y_shoulder))

    '''Starting derivation for upper and lower arm'''
    length_wrist_to_gripcenter = 0.0601+0.075

    z_world_to_upper = 0.1025 + 0.0165
    GL_shoulder_to_upper = 0.0306 #GL -> Ground length, specifically xy plane
    GL_shoulder_to_gripcenter = np.sqrt(x**2+(y-y_shoulder)**2)
    z_wrist = z - length_wrist_to_gripcenter*np.sin(pitch)

    ### Define 2D-2R problem
    x_2R_1 = GL_shoulder_to_gripcenter-GL_shoulder_to_upper-length_wrist_to_gripcenter*np.cos(pitch) #1: ThetaShoulder1
    x_2R_2 = -(GL_shoulder_to_gripcenter+GL_shoulder_to_upper+length_wrist_to_gripcenter*np.cos(pitch)) #2: ThetaShoulder2
    y_2R = z_wrist-z_world_to_upper

    ## Define lengths of segments and their offsets
    upper_x_length = 0.028
    upper_y_length = 0.11257
    upper_length = np.sqrt(upper_x_length**2 + upper_y_length**2)
    upper_alpha = np.arctan2(upper_y_length,upper_x_length) #Converting to axis aligned with joint

    lower_x_length = 0.1349
    lower_y_length = 0.0052
    lower_length = np.sqrt(lower_x_length**2 + lower_y_length**2)
    lower_beta = np.pi/2 - np.arctan2(upper_x_length,upper_y_length) - np.arctan2(lower_y_length,lower_x_length) #Converting to axis aligned with joint

    '''Direct pitch, Direct thetashoulder, elbow down'''
    # Cosine from the geometry
    cos_equiv_1 = ((x_2R_1)**2+(y_2R)**2-upper_length**2-lower_length**2)/(2*upper_length*lower_length)
    # Sin can be +- -> gives elbow up and down solution
    sin_equiv_1_1 = np.sqrt(1-cos_equiv_1**2)
    #Find theta2
    theta2_1_1 = np.arctan2(sin_equiv_1_1, cos_equiv_1)
    # Calculate Theta_lower
    theta_lower_1_1 = theta2_1_1 + lower_beta # Lower arm joint
    # Use method to find theta 1 (Theta_1 is a substitute to align angle with links for derivation)
    K1_1_1 = upper_length+lower_length*np.cos(theta2_1_1) ### Define factors for theta lower
    K2_1_1 = lower_length*np.sin(theta2_1_1)
    theta1_1_1 = np.arctan2(y_2R,x_2R_1)-np.arctan2(K2_1_1,K1_1_1)
    #Calculate Theta_Upper
    theta_upper_1_1 = theta1_1_1 - upper_alpha
    #Calculate Theta_Wrist
    theta_wrist_1_1 = pitch - theta_upper_1_1 - theta_lower_1_1  # Wrist joint

    '''Direct pitch, Direct thetashoulder, elbow up'''
    # Redo for the other sin
    sin_equiv_1_2 = -np.sqrt(1-cos_equiv_1**2)
    theta2_1_2 = np.arctan2(sin_equiv_1_2, cos_equiv_1)
    theta_lower_1_2 = theta2_1_2 + lower_beta # Lower arm joint
    K1_1_2 = upper_length+lower_length*np.cos(theta2_1_2)
    K2_1_2 = lower_length*np.sin(theta2_1_2)
    theta1_1_2 = np.arctan2(y_2R,x_2R_1)-np.arctan2(K2_1_2,K1_1_2)
    theta_upper_1_2 = theta1_1_2 - upper_alpha # Upper arm joint
    theta_wrist_1_2 = pitch - theta_upper_1_2 - theta_lower_1_2  # Wrist joint

    '''Direct pitch, Indirect thetashoulder, elbow down'''
    # Redo but for different theta_shoulder
    cos_equiv_2 = ((x_2R_2) ** 2 + (y_2R) ** 2 - upper_length ** 2 - lower_length ** 2) / (2 * upper_length * lower_length)

    sin_equiv_2_1 = np.sqrt(1-cos_equiv_2**2)
    theta2_2_1 = np.arctan2(sin_equiv_2_1,cos_equiv_2)
    theta_lower_2_1 = theta2_2_1 + lower_beta # Lower arm joint
    K1_2_1 = upper_length+lower_length*np.cos(theta2_2_1)
    K2_2_1 = lower_length*np.sin(theta2_2_1)
    theta1_2_1 = np.arctan2(y_2R,x_2R_2)-np.arctan2(K2_2_1,K1_2_1)
    theta_upper_2_1 = theta1_2_1 - upper_alpha # Upper arm joint
    theta_wrist_2_1 = pitch - theta_upper_2_1 - theta_lower_2_1  # Wrist joint

    '''Direct pitch, Indirect thetashoulder, elbow up'''
    sin_equiv_2_2 = -np.sqrt(1-cos_equiv_2**2)
    theta2_2_2 = np.arctan2(sin_equiv_2_2,cos_equiv_2)
    theta_lower_2_2 = theta2_2_2 + lower_beta # Lower arm joint
    K1_2_2 = upper_length+lower_length*np.cos(theta2_2_2)
    K2_2_2 = lower_length*np.sin(theta2_2_2)
    theta1_2_2 = np.arctan2(y_2R,x_2R_2)-np.arctan2(K2_2_2,K1_2_2)
    theta_upper_2_2 = theta1_2_2 - upper_alpha # Upper arm joint
    theta_wrist_2_2 = pitch - theta_upper_2_2 - theta_lower_2_2  # Wrist joint

    # Gripper roll joint
    theta_gripper = roll

    #Define vector
    q_vector_1_1 = np.array([theta_shoulder_1,theta_upper_1_1,theta_lower_1_1,theta_wrist_1_1,theta_gripper]) #Direct Thetashoulder, upper/lower/theta elbow ...
    q_vector_1_2 = np.array([theta_shoulder_1, theta_upper_1_2, theta_lower_1_2, theta_wrist_1_2, theta_gripper]) #Direct Thetashoulder, upper/lower/theta elbow ...
    q_vector_2_1 = np.array([theta_shoulder_2,theta_upper_2_1,theta_lower_2_1,theta_wrist_2_1,theta_gripper]) #Direct Thetashoulder, upper/lower/theta elbow ...
    q_vector_2_2 = np.array([theta_shoulder_2, theta_upper_2_2, theta_lower_2_2, theta_wrist_2_2, theta_gripper]) #Direct Thetashoulder, upper/lower/theta elbow ...

    return q_vector_1_1, q_vector_1_2, q_vector_2_1, q_vector_2_2

def JointFeasibilityCheck(q_vector):
    Feasibility = True
    # Shoulder
    if q_vector[0] < RC.shoulder_joint_limits[0] or q_vector[0]>RC.shoulder_joint_limits[1] or np.isnan(q_vector[0]):
        Feasibility = False
    elif q_vector[1] < RC.upper_joint_limits[0] or q_vector[1]>RC.upper_joint_limits[1] or np.isnan(q_vector[1]):
        Feasibility = False
    elif q_vector[2] < RC.lower_joint_limits[0] or q_vector[2]>RC.lower_joint_limits[1] or np.isnan(q_vector[2]):
        Feasibility = False
    # elif q_vector[3] < RC.wrist_joint_limits[0] or q_vector[3]>RC.wrist_joint_limits[1] or q_vector[3] == np.nan:
    #     Feasibility = False
    # elif q_vector[4] < RC.gripper_joint_limits[0] or q_vector[4]>RC.gripper_joint_limits[1] or q_vector[4] == np.nan:
    #     Feasibility = False

    return Feasibility

def SelectJointVector(q_list,old_q):
    FeasibileVectorList = []

    for q in q_list:
        if JointFeasibilityCheck(q) == True:
            FeasibileVectorList.append(q)

    if len(FeasibileVectorList) > 0:
        DiffArray = np.zeros(len(FeasibileVectorList))
        for i in range(len(FeasibileVectorList)):
            DiffArray[i] = np.linalg.norm((FeasibileVectorList[i]-old_q)**2)

        newQ_index = np.argsort(DiffArray == min(DiffArray))[0]

        return FeasibileVectorList[newQ_index], True
    else:
        return old_q, False



if __name__ == "__main__":
    q_home = [np.deg2rad(0), np.deg2rad(105),np.deg2rad(-70), np.deg2rad(-60),np.deg2rad(0)]
    old_q = q_home
    TrajectoryLength = len(TT.cartesian_trajectory)
    q_history = np.zeros((TrajectoryLength+1,len(q_home)))
    for i in range(len(TT.cartesian_trajectory)):
        q_history[i] = old_q
        print(f"\nTrajectory waypoint {i}")
        valid = False
        pitch_iter = 0
        Trajectory = TT.cartesian_trajectory[i]
        while not valid and pitch_iter < 20:
            Trajectory[3] = Trajectory[3]+(-1)**(pitch_iter)*0.025*np.pi*pitch_iter
            pitch_iter += 1
            q_vector_1, q_vector_2, q_vector_3, q_vector_4 = InverseKinematics(Trajectory)
            chosen_q, valid = SelectJointVector([q_vector_1,q_vector_2,q_vector_3,q_vector_4],old_q=old_q)
        print(f'Feasible solution found: {valid}')
        print(f'After {pitch_iter} iterations')
        print(f'Chosen q: {chosen_q}')
        # plot_robot_position(chosen_q)
        old_q = chosen_q

    q_history[-1]=chosen_q

    fig = plt.figure(figsize=(6, 6), dpi=125)
    ax = fig.add_subplot(111, projection='3d')
    cartesian_shape = TT.cartesian_trajectory

    def update(i):
        TJ0, TJ1, TJ2, TJ3, TJ4, TJ5, TJ6 = ForwardKinematicsFullOutput_PreRotationFrame(q_history[i])

        x, y, z = XYZ_from_T(TJ6)[0, 0], XYZ_from_T(TJ6)[1, 0], XYZ_from_T(TJ6)[2, 0]
        pitch = q_history[i,1] + q_history[i,2] + q_history[i,3]
        roll = q_history[i,4]

        state = np.array([x, y, z, pitch, roll])
        np.set_printoptions(suppress=True)
        # print(f'EE state: {state}')
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
        ax.cla()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label="Joint", color="black")
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="black")
        ax.scatter(X_points[:, 0], X_points[:, 1], X_points[:, 2], label="Frame X-axis", color="red")
        ax.scatter(Y_points[:, 0], Y_points[:, 1], Y_points[:, 2], label="Frame Y-axis", color="green")
        ax.scatter(Z_points[:, 0], Z_points[:, 1], Z_points[:, 2], label="Frame Z-axis", color="blue")
        ax.scatter(cartesian_shape[:, 0], cartesian_shape[:, 1], cartesian_shape[:, 2], color="orange")

        plt.axis('equal')
        ax.set_xlim(-0.3, 0.5)
        ax.set_ylim(-0.3, 0.5)
        ax.set_zlim(0, 0.5)

        ax.set_title(f"EE: {state},\n q: {q_history[i]}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # plt.show()
        np.set_printoptions(suppress=False)
        np.set_printoptions(precision=8)

    ani = FuncAnimation(fig, update, frames=len(q_history), interval=1000, repeat=True,repeat_delay=3000)

    plt.show()

            # Feasible, q_vector = SelectJointVector(q_vector_1, q_vector_2)
            #
            # if Feasible:
            #     plot_robot_position(q_vector)
            # else:
            #     print("Unfeasible position, moving on")

        # point1 = [0.2, 0.2, 0.2, 1.57, 0.0]
        # point2 = [0.2, 0.1, 0.4, 0.0, 1.57]
        # point3 = [0.0, 0.0, 0.45, 0.785, 0.785]
        # point4 = [0.0, 0.0, 0.07, 3.141, 0.0]
        # point5 = [0.0, 0.0452, 0.45, 0.785, 3.141]
        #
        #
        # point = point1
        #
        #
        # q_vector_1, q_vector_2, q_vector_3, q_vector_4 = InverseKinematics(point)