import numpy as np
import matplotlib.pyplot as plt
from DirectForwardKinematics import ForwardKinematics, plot_robot_position
import RobotConstants as RC
import TrajectoryTest as TT

def InverseKinematics(EE_state_vector):
    x,y,z,pitch,roll = EE_state_vector

    # Robot joint positions
    y_shoulder = 0.0452

    # Since this is the only "yaw" joint rotation
    theta_shoulder = np.arctan2(-x,y-y_shoulder) #shoulder joint

    length_wrist_to_gripcenter = 0.0601+0.075

    z_world_to_upper = 0.1025 + 0.0165
    GL_shoulder_to_upper = 0.0306 #GL -> Ground length, specifically xy plane
    GL_shoulder_to_gripcenter = np.sqrt(x**2+(y-y_shoulder)**2)
    z_wrist = z - length_wrist_to_gripcenter*np.sin(pitch)

    ### Define 2D-2R problem
    x_2R = GL_shoulder_to_gripcenter-GL_shoulder_to_upper-length_wrist_to_gripcenter*np.cos(pitch)
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

    #Find theta upper
    cos_equiv = ((x_2R)**2+(y_2R)**2-upper_length**2-lower_length**2)/(2*upper_length*lower_length)
    sin_equiv_1 = np.sqrt(1-cos_equiv**2)
    sin_equiv_2 = -np.sqrt(1-cos_equiv**2)

    theta2_1 = np.arctan2(sin_equiv_1,cos_equiv)
    theta2_2 = np.arctan2(sin_equiv_2,cos_equiv)

    theta_lower_1 = theta2_1 + lower_beta
    theta_lower_2 = theta2_2 + lower_beta

    ### Define factors for theta lower
    K1_1 = upper_length+lower_length*np.cos(theta2_1)
    K2_1 = lower_length*np.sin(theta2_1)

    K1_2 = upper_length+lower_length*np.cos(theta2_2)
    K2_2 = lower_length*np.sin(theta2_2)

    theta1_1 = np.arctan2(y_2R,x_2R)-np.arctan2(K2_1,K1_1)  # Upper arm joint
    theta1_2 = np.arctan2(y_2R,x_2R)-np.arctan2(K2_2,K1_2) # Lower arm  joint

    theta_upper_1 = theta1_1 - upper_alpha
    theta_upper_2 = theta1_2 - upper_alpha

    # Wrist angle
    theta_wrist_1 = pitch - theta_upper_1 - theta_lower_1  # Wrist joint
    theta_wrist_2 = pitch - theta_upper_2 - theta_lower_2  # Wrist joint

    # Gripper roll joint
    theta_gripper = roll

    #Define vector
    q_vector_1 = np.array([theta_shoulder,theta_upper_1,theta_lower_1,theta_wrist_1,theta_gripper])
    q_vector_2 = np.array([theta_shoulder, theta_upper_2, theta_lower_2, theta_wrist_2, theta_gripper])

    return q_vector_1, q_vector_2

def JointFeasibilityCheck(q_vector):
    Feasibility = True
    # Shoulder
    if q_vector[0] < RC.shoulder_joint_limits[0] or q_vector[0]>RC.shoulder_joint_limits[1]:
        Feasibility = False
    elif q_vector[1] < RC.upper_joint_limits[0] or q_vector[1]>RC.upper_joint_limits[1]:
        Feasibility = False
    elif q_vector[2] < RC.lower_joint_limits[0] or q_vector[2]>RC.lower_joint_limits[1]:
        Feasibility = False
    elif q_vector[3] < RC.wrist_joint_limits[0] or q_vector[3]>RC.wrist_joint_limits[1]:
        Feasibility = False
    elif q_vector[4] < RC.gripper_joint_limits[0] or q_vector[4]>RC.gripper_joint_limits[1]:
        Feasibility = False
    else:
        Feasibility = True

    return Feasibility

def SelectJointVector(first_q, second_q):
    Feasibility_1 = JointFeasibilityCheck(first_q)
    Feasibility_2 = JointFeasibilityCheck(second_q)
    Feasible = True

    if Feasibility_1:
        return Feasible, first_q
    elif Feasibility_2:
        return Feasible, second_q
    else:
        Feasible = False
        return Feasible, first_q

if __name__ == '__main__':
    # State: X, Y, Z, pitch roll
    # points = np.array([[0.2, 0.2, 0.2, 1.57, 0.0],
    #                    [0.2, 0.1, 0.4, 0.0, 1.57],
    #                    [0.0, 0.0, 0.45, 0.785, 0.785],
    #                    [0.0, 0.0, 0.07, 3.141, 0.0],
    #                    [0.0, 0.0452, 0.45, 0.785, 3.141]])

    for i in range(len(TT.cartesian_trajectory)):
        valid = False
        q_vector_1, q_vector_2 = InverseKinematics(TT.cartesian_trajectory[i])

        Feasible, q_vector = SelectJointVector(q_vector_1, q_vector_2)

        if Feasible:
            plot_robot_position(q_vector)
        else:
            print("Unfeasible position, moving on")

        plt.show()

    # print(f'Waypoint: {point}')
    #
    # print(f'\nq1 state: {q_vector_1}')
    # plot_robot_position(q_vector_1)
    #
    # print(f'\nq2 state: {q_vector_2}')
    # plot_robot_position(q_vector_2)
