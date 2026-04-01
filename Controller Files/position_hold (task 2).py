import rclpy
import numpy as np
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

'''This file was used to place the robot in the positions for task 2'''

class ExampleTraj(Node):

    def __init__(self):
        super().__init__('example_trajectory')

        # Home matches URDF/sim: 0, 105°, -70°, -60°, 0 deg
        self._HOME = [np.deg2rad(0), np.deg2rad(105),
                      np.deg2rad(-70), np.deg2rad(-60),
                      np.deg2rad(0)]
        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        timer_period = 0.04  # seconds
        self._timer = self.create_timer(timer_period, self.timer_callback)
        self.old_q = self._HOME
        self.jaw = 0.15*np.pi

        '''Input hold position'''
        self.hold_position = [-0.91211132,-1.80315694,3.45338614,-1.64943287,2.22079633]

    def timer_callback(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        point = JointTrajectoryPoint()
        point.positions = [self.hold_position[0],
                           self.hold_position[1],
                           self.hold_position[2],
                           self.hold_position[3],
                           self.hold_position[4],
                           0.0] #self.jaw
        msg.points = [point]

        self._publisher.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)

    example_traj = ExampleTraj()

    rclpy.spin(example_traj)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    example_traj.destroy_node()
    rclpy.shutdown()

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
    shoulder_joint_limits = np.array([-1.96, 2.15])  # rad
    upper_joint_limits = np.array([-1.99, 1.67])  # rad
    lower_joint_limits = np.array([-1.62, 1.74])  # rad
    wrist_joint_limits = np.array([-1.82, 1.80])  # rad
    gripper_joint_limits = np.array([-2.93, 2.92])  # rad
    # Shoulder
    if q_vector[0] < shoulder_joint_limits[0] or q_vector[0]>shoulder_joint_limits[1] or np.isnan(q_vector[0]):
        Feasibility = False
    elif q_vector[1] < upper_joint_limits[0] or q_vector[1]>upper_joint_limits[1] or np.isnan(q_vector[1]):
        Feasibility = False
    elif q_vector[2] < lower_joint_limits[0] or q_vector[2]>lower_joint_limits[1] or np.isnan(q_vector[2]):
        Feasibility = False
    elif q_vector[3] < wrist_joint_limits[0] or q_vector[3]>wrist_joint_limits[1] or q_vector[3] == np.nan:
        Feasibility = False
    elif q_vector[4] < gripper_joint_limits[0] or q_vector[4]>gripper_joint_limits[1] or q_vector[4] == np.nan:
        Feasibility = False

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

if __name__ == '__main__':
    main()
