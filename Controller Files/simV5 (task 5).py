
import rclpy
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ── GLOBAL TUNING ──────────────────────────────────────────
SPEED_TRAVEL = 0.08  # Lower is slower (0.01 = very slow, 0.1 = fast) - For moving
SPEED_PRECISION = 0.06   # Lower is slower - For precision tasks
WAIT_TIME = 0.6    # Seconds to wait for jaw opening/closing

class PickAndPlaceNode(Node):

    def __init__(self):
        super().__init__('pick_and_place_node')

        # ── Configuration ──────────────────────────────────────────
        self._HOME_Q = [np.deg2rad(0), np.deg2rad(105), np.deg2rad(-70), np.deg2rad(-60), np.deg2rad(-90)]
        
        # Cube / Gripper geometry
        self._CUBE_SIZE = 0.036    # 4 cm cube
        self._GRASP_MARGIN = 0.005    # grip slightly tighter (5 mm)
        self._FINGER_LENGTH = 0.07    # pivot - fingertip length (meters)

        # URDF joint limits 
        self._GRIPPER_LOWER = -0.2
        self._GRIPPER_UPPER = 2.0

        # Computing the grasp command value
        target_width = self._CUBE_SIZE - self._GRASP_MARGIN
        theta = 2 * np.arcsin(target_width / (2 * self._FINGER_LENGTH))

        self._GRASP_VAL = (theta - self._GRIPPER_LOWER) / (self._GRIPPER_UPPER - self._GRIPPER_LOWER)



        # Poses: [x, y, z, pitch, roll]
        self._PICK_POSE  = [-0.03, 0.12, 0.02, -np.pi/2, -1.5707]
        self._PLACE_POSE = [0.10, 0.11, 0.02, -np.pi/2, -1.5707]
        self.calibration = np.array([-0.033, -0.21, -0.06, 0.01, 0.02])   #needed to tune for errors in the physical robot setup
        self._TRAVEL_Z   = 0.20  # Final height for safety
       

        self._APPROACH_OFFSET = 0.04   # safe height above stacks where the robot moves before descending

        self._STACK_HEIGHT = 0.02   #height of each object
        self._MAX_OBJECTS = 6       #maximum number of objects 

        #self._pick_stack_level = self._MAX_OBJECTS - 1   #pick stack starts full
        self._pick_stack_level = 0
        self._place_stack_level = 0       # place stack starts empty

        self._just_changed_state = False

        # ── State Machine Variables ──────────────────────────────────────
        self._state = "APPROACH_PICK"
        self._wait_start_time = None
        self._old_q = np.array(self._HOME_Q)
        self._gripper_val = 0.5 # Start Open
        self._target_ik_q = np.array(self._HOME_Q) # Stores the current goal IK
        
        self._active_pick_pose = None

        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self._timer = self.create_timer(0.04, self.timer_callback)
        
        self.get_logger().info(f'Slow-Motion Node Started.')


    def _get_current_pick_pose(self):
        z_pick = self._PICK_POSE[2] + self._pick_stack_level * self._STACK_HEIGHT
        return [self._PICK_POSE[0], self._PICK_POSE[1], z_pick, self._PICK_POSE[3], -1.5707]
    

    def _get_current_place_pose(self):
        z_place = self._PLACE_POSE[2] + self._place_stack_level * self._STACK_HEIGHT
        return [self._PLACE_POSE[0], self._PLACE_POSE[1], z_place, self._PLACE_POSE[3] , -1.5707]
    

    def _get_pick_approach_pose(self):

        pick_pose = self._get_current_pick_pose()
        place_pose = self._get_current_place_pose()         #added for pick approach height

        return [
            pick_pose[0],
            pick_pose[1],
            place_pose[2] + self._APPROACH_OFFSET - self._STACK_HEIGHT,
            pick_pose[3],
            pick_pose[4]
        ]

    
    def _get_place_approach_pose(self):
        place_pose = self._get_current_place_pose()
        
        return [
            place_pose[0],
            place_pose[1],
            place_pose[2] + self._APPROACH_OFFSET,
            place_pose[3],
            place_pose[4]
        ]


    def timer_callback(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        
        
        target_cartesian = np.zeros(5)

        # ── STATE LOGIC: Setting the Cartesian Goal ──────────────────────
        if self._state == "APPROACH_PICK":
            self._gripper_val = 0.4 
            target_cartesian = self._get_pick_approach_pose()
        
        elif self._state == "DESCEND_TO_PICK":
            target_cartesian = self._get_current_pick_pose()


        elif self._state == "GRASPING":
            target_cartesian = self._get_current_pick_pose()
            elapsed = (now - self._wait_start_time).nanoseconds * 1e-9

            t = min(elapsed, WAIT_TIME)

            # Cosine easing for gripper movement

            s = 0.5 * (1 + np.cos(np.pi * t / WAIT_TIME))

            start = 0.4
            end = self._GRASP_VAL

            # Smooth slow jaw closing
            self._gripper_val = end + (start - end ) * s
            if elapsed >= WAIT_TIME:
                self._gripper_val = end
                self._state = "LIFTING"
                self._just_changed_state = True
                self.get_logger().info("Object secured. Lifting...")

        elif self._state == "LIFTING":
            pick_pose = self._active_pick_pose
            place_pose = self._get_current_place_pose()
            target_cartesian = [
                pick_pose[0],
                pick_pose[1],
                place_pose[2] + self._APPROACH_OFFSET,
                pick_pose[3],
                pick_pose[4]
            ]
            

        elif self._state == "TRAVELING_ONE":
            target_cartesian = self._get_place_approach_pose()
            

        elif self._state == "LOWERING":
            target_cartesian = self._get_current_place_pose()
            

        elif self._state == "RELEASING":
            target_cartesian = self._get_current_place_pose()
            elapsed = (now - self._wait_start_time).nanoseconds * 1e-9

            t = min(elapsed, WAIT_TIME)

            s = 0.5 * (1 - np.cos(np.pi * t / WAIT_TIME))

            start = self._GRASP_VAL
            end = 0.4

            # Smooth slow jaw opening
            self._gripper_val = start + (end - start) * s
            if elapsed >= WAIT_TIME:
                self._gripper_val = end
                self._place_stack_level += 1
                self.get_logger().info(f"Pick Stack level : {self._pick_stack_level}, Place stack level : {self._place_stack_level}")
                if self._place_stack_level >= self._MAX_OBJECTS:
                    self.get_logger().info("Stacking complete. All objects placed.")
                    self._state = "DONE"
                    self._just_changed_state = True
                    return
                self._state = "RESET_LIFT"
                self._just_changed_state = True

        elif self._state == "RESET_LIFT":
            target_cartesian = [
                self._PLACE_POSE[0], 
                self._PLACE_POSE[1], 
                self._APPROACH_OFFSET + self._place_stack_level * self._STACK_HEIGHT,           
                self._PLACE_POSE[3], 
                self._PLACE_POSE[4]
            ]
        
        elif self._state == "DONE":
            target_cartesian = [
                self._PLACE_POSE[0], 
                self._PLACE_POSE[1], 
                self._TRAVEL_Z,           
                self._PLACE_POSE[3], 
                self._PLACE_POSE[4]
            ]


        if self._state in ["APPROACH_PICK", "DESCEND_TO_PICK", "GRASPING", "LOWERING", "RELEASING"]:
            target_cartesian[3] = -np.pi/2
        else:
            target_cartesian[3] = -np.pi/2

        # ROLL COMPENSATION : keep gripper world-orientation fixed

        y_shoulder = 0.0452
        theta_shoulder = np.arctan2(target_cartesian[0], target_cartesian[1] - y_shoulder)
        desired_world_roll = -1.5707                     # the orientation we want to hold
        target_cartesian[4] = desired_world_roll - theta_shoulder 

        precision_states = ["DESCEND_TO_PICK", "GRASPING", "LOWERING", "RELEASING"]

        if self._state in precision_states:
            current_move_speed = SPEED_PRECISION
        else:
            current_move_speed = SPEED_TRAVEL



        # ── IK ENGINE: Moving Joints slowly toward target ───────────────
        valid = False
        pitch_iter = 0
        base_pitch = target_cartesian[3]
        while not valid and pitch_iter < 20:
            target_cartesian[3] = -np.pi/2 + (-1)**(pitch_iter) * 0.025 * np.pi * pitch_iter
            pitch_iter += 1
            q_results = InverseKinematics(target_cartesian)
            chosen_q, valid = SelectJointVector(q_results, old_q=self._old_q)
            chosen_q = chosen_q + self.calibration
        if valid:
            self._target_ik_q = chosen_q # Update what we are trying to reach
            # The 'Speed' happens here:
            
            step_q = self._old_q + (self._target_ik_q - self._old_q) * current_move_speed 
            self._publish_joints(step_q, self._gripper_val, msg)
            self._old_q = step_q
        else:
            self.get_logger().error(f"IK FAIL in state {self._state}")
        
        if self._reached_target(target_cartesian):
            if self._state == "APPROACH_PICK":
                self._state = "DESCEND_TO_PICK"
                self._just_changed_state = True
                self._wait_start_time = now
                self.get_logger().info("Reached above target. Descending to grasp...")  
            
            elif self._state == "DESCEND_TO_PICK":
                self._active_pick_pose = self._get_current_pick_pose()
                self.get_logger().info(f"Pick surface reached. Starting grasp | pick_level = {self._pick_stack_level}")
                self._state = "GRASPING"
                self._just_changed_state = True
                self._wait_start_time = now

            elif self._state == "LIFTING":
                self._state = "TRAVELING_ONE"
                self._just_changed_state = True
                self.get_logger().info("Safe height reached. Traveling...")

            elif self._state == "TRAVELING_ONE":
                self._state = "LOWERING"
                self._just_changed_state = True
                self.get_logger().info("Travel complete.")

            elif self._state == "LOWERING":
                self._state = "RELEASING"
                self._just_changed_state = True
                self._wait_start_time = now
                self.get_logger().info("Ready to drop.")

            elif self._state == "RESET_LIFT":
                self._state = "APPROACH_PICK"
                self._just_changed_state = True
                self.get_logger().info("Cycle restarting.")          

    def _reached_target(self, target_cartesian):
        # Checks if the arm joints have arrived at the required IK solution
        if self._just_changed_state:
            self._just_changed_state = False
            return False
        current_xyz = ForwardKinematics(self._old_q)

        target_xyz = np.array(target_cartesian[:3])

        dist = np.linalg.norm(current_xyz - target_xyz)

        return dist < 0.002
        


    def _compute_move_time(self, q_target):
        dist = np.linalg.norm(self._old_q - q_target)
        speed = 1.5          # rad/sec (tune later)
        return max(0.4, dist/speed)

    def _publish_joints(self, q, gripper, msg, duration = 1.0):
        point = JointTrajectoryPoint()
        point.positions = [float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(gripper)]

        point.time_from_start = Duration(sec = int(duration),nanosec = int((duration % 1.0) * 1e9))

        msg.points = [point]
        self._publisher.publish(msg)

# ====================================================================== #
# IK / FEASIBILITY HELPERS
# ====================================================================== #

def InverseKinematics(EE_state_vector):
    x,y,z,pitch,roll = EE_state_vector

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
    # i hate floating points
    if cos_equiv_1 > 1:
        cos_equiv_1 = 1
    if cos_equiv_1 < -1:
        cos_equiv_1 = -1

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
    # i hate floating points
    if cos_equiv_2 > 1:
        cos_equiv_2 = 1
    if cos_equiv_2 < -1:
        cos_equiv_2 = -1

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

    return q_vector_1_1, q_vector_1_2, q_vector_2_1, q_vector_2_2 #1: Direct, Elbow down, 2: Direct Elbow up, 3: Indirect, Elbow down, 4: Indirect, Elbow up

def ForwardKinematics(q):
    """Calculates Cartesian XYZ from joint angles based on the arm's specific geometry."""
    theta_shoulder, theta_upper, theta_lower, theta_wrist, theta_gripper = q
    
    # Arm Geometry Constants (Pulled directly from your IK solver)
    y_shoulder = 0.0452
    length_wrist_to_gripcenter = 0.1351  # 0.0601 + 0.075
    z_world_to_upper = 0.119             # 0.1025 + 0.0165
    GL_shoulder_to_upper = 0.0306
    
    # Upper Link
    upper_x_length = 0.028
    upper_y_length = 0.11257
    upper_length = np.sqrt(upper_x_length**2 + upper_y_length**2)
    upper_alpha = np.arctan2(upper_y_length, upper_x_length)
    
    # Lower Link
    lower_x_length = 0.1349
    lower_y_length = 0.0052
    lower_length = np.sqrt(lower_x_length**2 + lower_y_length**2)
    lower_beta = np.pi/2 - np.arctan2(upper_x_length, upper_y_length) - np.arctan2(lower_y_length, lower_x_length)
    
    # Reverse the angles into the 2R planar system
    theta1 = theta_upper + upper_alpha
    theta2 = theta_lower - lower_beta
    pitch = theta_wrist + theta_upper + theta_lower
    
    # Calculate 2R Planar Coordinates
    x_2R = upper_length * np.cos(theta1) + lower_length * np.cos(theta1 + theta2)
    y_2R = upper_length * np.sin(theta1) + lower_length * np.sin(theta1 + theta2)
    
    # Map back to World Z
    z_wrist = y_2R + z_world_to_upper
    z = z_wrist + length_wrist_to_gripcenter * np.sin(pitch)
    
    # Map back to World X and Y
    R = x_2R + GL_shoulder_to_upper + length_wrist_to_gripcenter * np.cos(pitch)
    x = -R * np.sin(theta_shoulder)
    y = R * np.cos(theta_shoulder) + y_shoulder
    
    return np.array([x, y, z])


def JointFeasibilityCheck(q):
    lims = [(-1.96, 2.15), (-1.99, 1.67), (-1.62, 1.74), (-1.82, 1.80), (-2.93, 2.92)]
    return all(lims[i][0] <= q[i] <= lims[i][1] for i in range(5))

def SelectJointVector(qs, old_q):
    feas = [q for q in qs if JointFeasibilityCheck(q)]
    if not feas: return old_q, False

    elbow_up = [q for q in feas if q[2] > -0.2]

    if elbow_up:
        feas = elbow_up

    return feas[np.argmin([np.linalg.norm(q - old_q) for q in feas])], True

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()