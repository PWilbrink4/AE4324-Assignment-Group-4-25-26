"""
task33_line_velocity.py
=======================
Task 3.3 — Jacobian velocity control on a TO-AND-FRO LINE trajectory along Y.

Trajectory:
  X = X_OFFSET          (fixed)
  Y oscillates linearly between Y_CENTER - HALF_RANGE and Y_CENTER + HALF_RANGE
  Z = Z_OFFSET          (fixed)
  pitch = -pi/2         (fixed downward)
  roll  = arctan2(Y, X_OFFSET)

  Because X_OFFSET = 0.0 and Y > 0 throughout, roll = pi/2 (constant)
  and roll_rate = 0.

  alpha sweeps 0 → 2π over one PERIOD (triangular wave):
    Seg 0  alpha ∈ [0,   π)   →  Y: Y_MIN → Y_MAX   (vy = +V_LINE)
    Seg 1  alpha ∈ [π, 2π)   →  Y: Y_MAX → Y_MIN   (vy = -V_LINE)

  where V_LINE = 2 * HALF_RANGE / (PERIOD / 2) = 4 * HALF_RANGE / PERIOD

Control mode : VELOCITY  (point.velocities — same as ExampleTraj)
Topic        : 'joint_cmds'  (JointTrajectory)
Rate         : 25 Hz

Run modes
---------
  python task33_line_velocity.py        -> simulation + plots
  ros2 run <pkg> task33_line_velocity   -> live ROS2 node
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

'''This file was used for the velocity control tasks and simulation
    Running with a connected robot results in it performing the trajectory,
    Running without the robot results in the simulation and the plots
'''

try:
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    ROS2_OK = True
except ImportError:
    ROS2_OK = False


# =============================================================================
#  1. ROTATION / TRANSLATION / HOMOGENEOUS HELPERS  (unchanged)
# =============================================================================

def RotationMatrix_X(angle):
    return np.array([
        [1, 0,               0              ],
        [0, np.cos(angle),  -np.sin(angle)  ],
        [0, np.sin(angle),   np.cos(angle)  ]
    ])

def RotationMatrix_Y(angle):
    return np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [ 0,             1, 0            ],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def RotationMatrix_Z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

def TranslationMatrix(x, y, z):
    return np.array([[x], [y], [z]])

def HomogeneousTransformation(rotation_matrix, translation_matrix):
    T = np.hstack((rotation_matrix, translation_matrix))
    T = np.vstack((T, np.array([[0, 0, 0, 1]])))
    return T


# =============================================================================
#  2. FORWARD KINEMATICS  (unchanged)
# =============================================================================

def _build_transforms(q_vector):
    theta_shoulder, theta_upper, theta_lower, theta_wrist, theta_gripper = q_vector
    Z0 = TranslationMatrix(0, 0, 0)

    T_world_base = HomogeneousTransformation(
        RotationMatrix_Z(np.pi), TranslationMatrix(0, 0, 0))
    T_base_shoulder = HomogeneousTransformation(
        RotationMatrix_X(0), TranslationMatrix(0, -0.0452, 0.0165))
    T_shoulder_upper = (
        HomogeneousTransformation(RotationMatrix_Z(theta_shoulder), Z0)
        @ HomogeneousTransformation(RotationMatrix_Y(-np.pi / 2),
                                    TranslationMatrix(0, -0.0306, 0.1025)))
    T_upper_lower = (
        HomogeneousTransformation(RotationMatrix_Z(theta_upper), Z0)
        @ HomogeneousTransformation(RotationMatrix_Z(0),
                                    TranslationMatrix(0.11257, -0.028, 0)))
    T_lower_wrist = (
        HomogeneousTransformation(RotationMatrix_Z(theta_lower), Z0)
        @ HomogeneousTransformation(RotationMatrix_Z(np.pi / 2),
                                    TranslationMatrix(0.0052, -0.1349, 0)))
    T_wrist_gripper = (
        HomogeneousTransformation(RotationMatrix_Z(theta_wrist), Z0)
        @ HomogeneousTransformation(RotationMatrix_Y(-np.pi / 2),
                                    TranslationMatrix(-0.0601, 0, 0)))
    T_gripper_grippercenter = (
        HomogeneousTransformation(RotationMatrix_Z(theta_gripper), Z0)
        @ HomogeneousTransformation(RotationMatrix_Z(0),
                                    TranslationMatrix(0, 0, 0.075)))
    return (T_world_base, T_base_shoulder, T_shoulder_upper,
            T_upper_lower, T_lower_wrist, T_wrist_gripper,
            T_gripper_grippercenter)


def ForwardKinematics(q_vector):
    (T_world_base, T_base_shoulder, T_shoulder_upper,
     T_upper_lower, T_lower_wrist, T_wrist_gripper,
     T_gripper_grippercenter) = _build_transforms(q_vector)
    return (T_world_base @ T_base_shoulder @ T_shoulder_upper
            @ T_upper_lower @ T_lower_wrist @ T_wrist_gripper
            @ T_gripper_grippercenter)


def XYZ_from_T(T):
    return (T @ np.array([0, 0, 0, 1]))[:3]


# =============================================================================
#  3. JOINT LIMITS & GEOMETRY CONSTANTS  (unchanged)
# =============================================================================

JOINT_NAMES = ['shoulder', 'upper', 'lower', 'wrist', 'gripper']

JOINT_LIMITS = [
    (-1.96,  2.15),   # shoulder
    (-1.67,  1.67),   # upper arm
    (-1.62,  1.74),   # lower arm
    (-1.82,  1.80),   # wrist
    (-2.93,  2.92),   # gripper
]

Y_SHOULDER           = 0.0452
Z_WORLD_TO_UPPER     = 0.1025 + 0.0165
GL_SHOULDER_TO_UPPER = 0.0306
UPPER_LENGTH         = np.sqrt(0.028**2  + 0.11257**2)
LOWER_LENGTH         = np.sqrt(0.1349**2 + 0.0052**2)
LENGTH_WRIST_TO_GC   = 0.0601 + 0.075


def JointFeasibilityCheck(q_vector):
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        val = q_vector[i]
        if np.isnan(val) or val < lo or val > hi:
            return False
    return True


def JointFeasibilityCheck_verbose(q_vector):
    violations = []
    for i, (name, (lo, hi)) in enumerate(zip(JOINT_NAMES, JOINT_LIMITS)):
        val = q_vector[i]
        if np.isnan(val):
            violations.append(f'{name}=NaN')
        elif val < lo:
            violations.append(
                f'{name}={np.degrees(val):.1f}deg < min {np.degrees(lo):.1f}deg')
        elif val > hi:
            violations.append(
                f'{name}={np.degrees(val):.1f}deg > max {np.degrees(hi):.1f}deg')
    return violations


def select_feasible_ik(candidates, reference_q):
    feasible = [q for q in candidates if JointFeasibilityCheck(q)]
    if not feasible:
        return None
    diffs = [np.linalg.norm(q - reference_q) for q in feasible]
    return feasible[int(np.argmin(diffs))]


# =============================================================================
#  4. INVERSE KINEMATICS  (unchanged)
# =============================================================================

def InverseKinematics(EE_state_vector):
    x, y, z, pitch, roll = EE_state_vector

    theta_shoulder    = np.arctan2(-x, y - Y_SHOULDER)
    GL_shoulder_to_gc = np.sqrt(x**2 + (y - Y_SHOULDER)**2)
    z_wrist           = z - LENGTH_WRIST_TO_GC * np.sin(pitch)
    x_2R = GL_shoulder_to_gc - GL_SHOULDER_TO_UPPER - LENGTH_WRIST_TO_GC * np.cos(pitch)
    y_2R = z_wrist - Z_WORLD_TO_UPPER

    upper_alpha = np.arctan2(0.11257, 0.028)
    lower_beta  = (np.pi / 2
                   - np.arctan2(0.028,  0.11257)
                   - np.arctan2(0.0052, 0.1349))

    cos_theta2 = np.clip(
        (x_2R**2 + y_2R**2 - UPPER_LENGTH**2 - LOWER_LENGTH**2)
        / (2 * UPPER_LENGTH * LOWER_LENGTH), -1.0, 1.0)

    results = []
    for sin_theta2 in (np.sqrt(1 - cos_theta2**2), -np.sqrt(1 - cos_theta2**2)):
        theta2      = np.arctan2(sin_theta2, cos_theta2)
        K1          = UPPER_LENGTH + LOWER_LENGTH * np.cos(theta2)
        K2          = LOWER_LENGTH * np.sin(theta2)
        theta1      = np.arctan2(y_2R, x_2R) - np.arctan2(K2, K1)
        theta_upper = theta1 - upper_alpha
        theta_lower = theta2 + lower_beta
        theta_wrist = pitch - theta_upper - theta_lower
        results.append(np.array([theta_shoulder, theta_upper,
                                  theta_lower, theta_wrist, roll]))
    return results[0], results[1]


# =============================================================================
#  5. JACOBIAN  (unchanged)
# =============================================================================

def Jacobian(q_vector):
    ts, tu, tl, tw, _ = q_vector
    tlu  = tl + tu
    tluw = tl + tu + tw

    reach = (  0.11257*np.sin(tu) + 0.0052*np.sin(tlu)
             - 0.028*np.cos(tu)   - 0.1349*np.cos(tlu)
             - 0.1351*np.cos(tluw) - 0.0306)
    arm_u = (  0.028*np.sin(tu)   + 0.1349*np.sin(tlu)
             + 0.1351*np.sin(tluw) + 0.11257*np.cos(tu)
             + 0.0052*np.cos(tlu))
    arm_l = (  0.1349*np.sin(tlu) + 0.1351*np.sin(tluw)
             + 0.0052*np.cos(tlu))
    arm_w =    0.1351*np.sin(tluw)
    dz_u  = ( -0.11257*np.sin(tu) - 0.0052*np.sin(tlu)
              + 0.028*np.cos(tu)  + 0.1349*np.cos(tlu)
              + 0.1351*np.cos(tluw))
    dz_l  = ( -0.0052*np.sin(tlu) + 0.1349*np.cos(tlu)
              + 0.1351*np.cos(tluw))
    dz_w  =    0.1351*np.cos(tluw)

    return np.array([
        [reach*np.cos(ts),   arm_u*np.sin(ts),  arm_l*np.sin(ts),  arm_w*np.sin(ts),  0],
        [reach*np.sin(ts),  -arm_u*np.cos(ts), -arm_l*np.cos(ts), -arm_w*np.cos(ts),  0],
        [0,                  dz_u,              dz_l,              dz_w,               0],
        [0,                  1,                 1,                 1,                  0],
        [0,                  0,                 0,                 0,                  1],
    ])


def damped_pinv(J, lam=0.008):
    n = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(n))


# =============================================================================
#  6. TO-AND-FRO LINE TRAJECTORY  (along Y axis)
#
#   X = X_OFFSET    (fixed throughout)
#   Z = Z_OFFSET    (fixed throughout)
#   Y oscillates linearly: triangular wave
#
#     Seg 0  alpha ∈ [0, π)   →  Y from Y_MIN to Y_MAX   vy = +V_LINE
#     Seg 1  alpha ∈ [π, 2π)  →  Y from Y_MAX to Y_MIN   vy = -V_LINE
#
#   V_LINE = 2*HALF_RANGE / (PERIOD/2) = 4*HALF_RANGE / PERIOD
#
#   pitch = -pi/2  (fixed downward)
#   roll  = arctan2(Y, X_OFFSET)
#           With X_OFFSET = 0.0 and Y > 0: roll = pi/2 (constant, roll_rate=0)
# =============================================================================

PERIOD     = 10.0
HALF_RANGE = 0.05           # half-stroke along Y  (total stroke = 0.10 m)
X_OFFSET   = 0.0            # fixed X (shoulder plane)
Y_CENTER   = 0.2            # centre of the stroke
Z_OFFSET   = 0.1            # fixed Z
PITCH      = -np.pi / 2     # fixed downward
OMEGA      = 2 * np.pi / PERIOD

Y_MIN = Y_CENTER - HALF_RANGE   # 0.15 m
Y_MAX = Y_CENTER + HALF_RANGE   # 0.25 m

# Constant Cartesian speed along Y on each stroke
_V_LINE = 4.0 * HALF_RANGE / PERIOD   # = 2*HALF_RANGE / (PERIOD/2)

# roll is constant because X = 0 and Y > 0 always
_ROLL_FIXED = np.arctan2(Y_CENTER, X_OFFSET)   # = pi/2


def line_pos(alpha):
    """
    EE pose at parametric angle alpha on the to-and-fro line.
    Returns [x, y, z, pitch, roll].

    Seg 0 (alpha ∈ [0, π)):   s = alpha / π,           Y = Y_MIN + 2*HALF_RANGE*s
    Seg 1 (alpha ∈ [π, 2π)):  s = (alpha - π) / π,     Y = Y_MAX - 2*HALF_RANGE*s
    """
    a = alpha % (2 * np.pi)
    if a < np.pi:                       # forward stroke
        s = a / np.pi
        y = Y_MIN + 2 * HALF_RANGE * s
    else:                               # return stroke
        s = (a - np.pi) / np.pi
        y = Y_MAX - 2 * HALF_RANGE * s

    roll = np.arctan2(y, X_OFFSET) if X_OFFSET != 0.0 else _ROLL_FIXED
    return np.array([X_OFFSET, y, Z_OFFSET, PITCH, roll])


def line_vel(alpha):
    """
    Analytical time-derivative of line_pos.

    vx = 0            (X fixed)
    vz = 0            (Z fixed)
    pitch_rate = 0    (pitch fixed)

    Seg 0: vy = +_V_LINE
    Seg 1: vy = -_V_LINE

    roll = arctan2(y, X_OFFSET)
    d/dt roll = X_OFFSET * vy / (X_OFFSET² + y²)
    With X_OFFSET = 0 → roll_rate = 0 (roll stays pi/2 throughout)
    """
    a  = alpha % (2 * np.pi)
    vy = +_V_LINE if a < np.pi else -_V_LINE

    y         = line_pos(alpha)[1]
    roll_rate = (X_OFFSET * vy / (X_OFFSET**2 + y**2)
                 if X_OFFSET != 0.0 else 0.0)

    return np.array([0.0, vy, 0.0, 0.0, roll_rate])


# =============================================================================
#  7. CONTROL PARAMETERS  (unchanged)
# =============================================================================

TIMER_PERIOD = 0.04    # seconds  (25 Hz)
MAX_QDOT     = 0.5     # rad/s
COND_LIMIT   = 200
LAMBDA       = 0.008

HOME = np.array([np.deg2rad(0),   np.deg2rad(105),
                 np.deg2rad(-70), np.deg2rad(-60), np.deg2rad(0)])


# =============================================================================
#  8. ROS2 NODE
# =============================================================================

if ROS2_OK:
    class LineVelocityNode(Node):
        """
        Jacobian velocity control on a to-and-fro Y-axis line trajectory.

        Each timer tick (25 Hz):
          1. alpha      = current angle on trajectory
          2. ee_vel     = line_vel(alpha)   piecewise-constant vy, all others zero
          3. J          = Jacobian(q_estimated)
          4. cond(J) checked → stop if > COND_LIMIT_REAL
          5. q_dot      = damped_pinv(J) * ee_vel
          6. q_est     += q_dot * dt        local integration
          7. publish q_dot in point.velocities
        """

        CMD_TOPIC       = 'joint_cmds'
        MAX_QDOT_REAL   = 0.5
        COND_LIMIT_REAL = 150
        LAMBDA_REAL     = 0.01

        def __init__(self):
            super().__init__('line_velocity_controller')

            HOME_q = HOME.copy()

            # Find feasible start configuration (beginning of forward stroke)
            q1, q2      = InverseKinematics(list(line_pos(0.0)))
            self._q     = select_feasible_ik([q1, q2], HOME_q)

            if self._q is None:
                for alpha_try in np.linspace(0, 2 * np.pi, 64):
                    q1, q2  = InverseKinematics(list(line_pos(alpha_try)))
                    self._q = select_feasible_ik([q1, q2], HOME_q)
                    if self._q is not None:
                        self._logical_time = alpha_try / OMEGA
                        break
                else:
                    self.get_logger().fatal('No feasible IK solution found!')
                    raise RuntimeError('No feasible start configuration')
            else:
                self._logical_time = 0.0

            cond0 = np.linalg.cond(Jacobian(self._q))
            self.get_logger().info(
                f'start q = {np.degrees(self._q).round(1)} deg  '
                f'cond(J) = {cond0:.1f}')
            if cond0 > 50:
                self.get_logger().warn('cond(J) > 50 at start — monitor carefully')

            self._publisher = self.create_publisher(
                JointTrajectory, self.CMD_TOPIC, 10)
            self._timer = self.create_timer(TIMER_PERIOD, self.timer_callback)
            self.get_logger().info(
                f'Publishing to "{self.CMD_TOPIC}" at {1/TIMER_PERIOD:.0f} Hz  '
                f'Y: {Y_MIN:.3f} ↔ {Y_MAX:.3f} m')

        def timer_callback(self):
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            self._logical_time += TIMER_PERIOD
            alpha = (self._logical_time % PERIOD) / PERIOD * 2 * np.pi

            ee_vel = line_vel(alpha)

            J    = Jacobian(self._q)
            cond = np.linalg.cond(J)

            if cond > self.COND_LIMIT_REAL:
                self.get_logger().error(
                    f'SINGULARITY: cond(J)={cond:.1f}. Publishing zero velocity.')
                self._publish_zero(msg)
                return

            Jp    = damped_pinv(J, self.LAMBDA_REAL)
            q_dot = Jp @ ee_vel

            mv = np.abs(q_dot).max()
            if mv > self.MAX_QDOT_REAL:
                q_dot *= self.MAX_QDOT_REAL / mv
                self.get_logger().warn(
                    f'velocity clamped {mv:.3f} -> {self.MAX_QDOT_REAL} rad/s')

            q_new = self._q + q_dot * TIMER_PERIOD
            for i, (lo, hi) in enumerate(JOINT_LIMITS):
                q_new[i] = float(np.clip(q_new[i], lo, hi))
            self._q = q_new

            point = JointTrajectoryPoint()
            for i in range(5):
                point.velocities.append(float(q_dot[i]))
            point.velocities.append(0.0)   # gripper closed

            msg.points = [point]
            self._publisher.publish(msg)

            seg = 0 if (alpha % (2 * np.pi)) < np.pi else 1
            y_now = line_pos(alpha)[1]
            self.get_logger().info(
                f'alpha={np.degrees(alpha):6.1f}deg  '
                f'seg={seg} ({"fwd" if seg==0 else "ret"})  '
                f'Y_ref={y_now:.4f}m  '
                f'cond={cond:5.1f}  '
                f'qdot_max={np.degrees(mv):5.1f}deg/s')

        def _publish_zero(self, msg):
            point = JointTrajectoryPoint()
            for _ in range(6):
                point.velocities.append(0.0)
            msg.points = [point]
            self._publisher.publish(msg)


# =============================================================================
#  9. SIMULATION
# =============================================================================

def simulate_line(n_laps=2):
    """Simulate n_laps full to-and-fro cycles."""
    n_steps = int(n_laps * PERIOD / TIMER_PERIOD)
    HOME_q  = HOME.copy()

    q1, q2 = InverseKinematics(list(line_pos(0.0)))
    q      = select_feasible_ik([q1, q2], HOME_q)
    if q is None:
        for a in np.linspace(0, 2 * np.pi, 64):
            q1, q2 = InverseKinematics(list(line_pos(a)))
            q      = select_feasible_ik([q1, q2], HOME_q)
            if q is not None:
                break
    if q is None:
        raise RuntimeError('No feasible start configuration')

    print(f'Start q (deg) : {np.degrees(q).round(2)}')
    print(f'Start EE      : {XYZ_from_T(ForwardKinematics(q)).round(4)} m')
    print(f'cond(J)       : {np.linalg.cond(Jacobian(q)):.2f}')
    print(f'Period        : {PERIOD} s  ({PERIOD/2:.2f} s per stroke)')
    print(f'Y range       : [{Y_MIN:.3f}, {Y_MAX:.3f}] m  '
          f'(centre={Y_CENTER:.3f} m  stroke={2*HALF_RANGE:.3f} m)\n')

    ea, ed, ql, qd_l, cl, erl, segl = [], [], [], [], [], [], []
    lt = 0.0

    for _ in range(n_steps):
        lt   += TIMER_PERIOD
        alpha  = (lt % PERIOD) / PERIOD * 2 * np.pi
        seg    = 0 if (alpha % (2 * np.pi)) < np.pi else 1
        ev     = line_vel(alpha)

        J    = Jacobian(q)
        cond = np.linalg.cond(J)
        cl.append(cond)

        if cond > COND_LIMIT:
            print(f'STOP: cond(J)={cond:.1f}')
            break

        Jp    = damped_pinv(J, LAMBDA)
        q_dot = Jp @ ev
        mv    = np.abs(q_dot).max()
        if mv > MAX_QDOT:
            q_dot *= MAX_QDOT / mv
        qd_l.append(q_dot.copy())

        q = q + q_dot * TIMER_PERIOD
        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            q[i] = float(np.clip(q[i], lo, hi))

        ee_now = XYZ_from_T(ForwardKinematics(q))
        ee_ref = line_pos(alpha)[:3]
        ea.append(ee_now); ed.append(ee_ref)
        ql.append(q.copy())
        erl.append(np.linalg.norm(ee_now - ee_ref) * 1000)
        segl.append(seg)

    ea   = np.array(ea);   ed   = np.array(ed)
    ql   = np.array(ql);   qd   = np.array(qd_l)
    cl   = np.array(cl);   er   = np.array(erl)
    segl = np.array(segl)
    t    = np.arange(len(ea)) * TIMER_PERIOD

    print(f'Max error  : {er.max():.2f} mm')
    print(f'Mean error : {er.mean():.2f} mm')
    print(f'Max cond(J): {cl.max():.2f}')
    return dict(t=t, ea=ea, ed=ed, ql=ql, qd=qd, cl=cl, er=er, segl=segl)


def plot_simulation(data):
    t    = data['t']
    ea   = data['ea'];   ed   = data['ed']
    ql   = data['ql'];   qd   = data['qd']
    cl   = data['cl'];   er   = data['er']
    segl = data['segl']
    cj   = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
    jn   = ['shoulder','upper','lower','wrist','gripper']

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f'Task 3.3 — Jacobian velocity control (to-and-fro Y-axis line)\n'
        f'Y: {Y_MIN:.3f} ↔ {Y_MAX:.3f} m  (stroke = {2*HALF_RANGE*1000:.0f} mm)  '
        f'X = {X_OFFSET} m  Z = {Z_OFFSET} m  fixed  '
        f'max_err = {er.max():.1f} mm  max_cond = {cl.max():.1f}',
        fontsize=11, fontweight='bold')

    # ---- Panel 1 : Y vs time (main trajectory view) ----
    ax1 = fig.add_subplot(2, 3, (1, 2))
    t_ref = np.linspace(0, t[-1], 2000)
    alpha_ref = (t_ref % PERIOD) / PERIOD * 2 * np.pi
    y_ref = np.array([line_pos(a)[1] for a in alpha_ref])
    ax1.plot(t_ref, y_ref, 'k--', lw=1.5, label='Desired Y(t)')

    fwd_mask = segl == 0
    ret_mask = segl == 1
    ax1.scatter(t[fwd_mask], ea[fwd_mask, 1], c='#4daf4a', s=6,
                zorder=3, label='Actual — forward (vy > 0)', alpha=0.9)
    ax1.scatter(t[ret_mask], ea[ret_mask, 1], c='#984ea3', s=6,
                zorder=3, label='Actual — return  (vy < 0)', alpha=0.9)
    ax1.axhline(Y_MIN,    color='steelblue', ls=':', lw=1, label=f'Y_MIN = {Y_MIN:.3f} m')
    ax1.axhline(Y_MAX,    color='tomato',    ls=':', lw=1, label=f'Y_MAX = {Y_MAX:.3f} m')
    ax1.axhline(Y_CENTER, color='gray',      ls=':', lw=1, label=f'Y_CEN = {Y_CENTER:.3f} m')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Y position (m)', fontsize=11)
    ax1.set_title(
        f'Y(t) — to-and-fro along Y axis  '
        f'(X = {X_OFFSET} m,  Z = {Z_OFFSET} m  fixed)',
        fontsize=10)
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)
    ax1.text(0.02, 0.07,
             f'max err = {er.max():.1f} mm\nmean err = {er.mean():.1f} mm',
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # ---- Panel 2 : tracking error ----
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.plot(t, er, color='crimson', lw=2)
    # Mark reversal points (multiples of PERIOD/2)
    for k in range(1, int(t[-1] / (PERIOD / 2)) + 1):
        ax2.axvline(k * PERIOD / 2, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax2.axhline(5, color='orange', ls='--', lw=1, label='5 mm limit')
    ax2.axhline(er.mean(), color='steelblue', ls='--', lw=1,
                label=f'mean = {er.mean():.1f} mm')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (mm)')
    ax2.set_title('Tracking error  (dotted = reversals)')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # ---- Panel 3 : joint angles ----
    ax3 = fig.add_subplot(2, 3, 4)
    for j, (nm, c) in enumerate(zip(jn, cj)):
        ax3.plot(t, np.degrees(ql[:, j]), color=c, lw=1.5, label=nm)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Angle (deg)')
    ax3.set_title('Joint angles')
    ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=0.3)

    # ---- Panel 4 : joint velocities ----
    ax4 = fig.add_subplot(2, 3, 5)
    qt  = np.arange(len(qd)) * TIMER_PERIOD
    for j, (nm, c) in enumerate(zip(jn, cj)):
        ax4.plot(qt, np.degrees(qd[:, j]), color=c, lw=1.5, label=nm)
    ax4.axhline( np.degrees(MAX_QDOT), color='red', ls='--', lw=1,
                 label='±limit')
    ax4.axhline(-np.degrees(MAX_QDOT), color='red', ls='--', lw=1)
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Vel (deg/s)')
    ax4.set_title('Joint velocities  q_dot = J⁺ · ẋ\n'
                  '(sign flip at each reversal)')
    ax4.legend(fontsize=7, ncol=2); ax4.grid(alpha=0.3)

    # ---- Panel 5 : Jacobian condition ----
    ax5 = fig.add_subplot(2, 3, 6)
    ct  = np.arange(len(cl)) * TIMER_PERIOD
    ax5.semilogy(ct, cl, color='steelblue', lw=2)
    ax5.axhline(COND_LIMIT, color='red',    ls='--', lw=1.5,
                label=f'Stop = {COND_LIMIT}')
    ax5.axhline(50, color='orange', ls='--', lw=1, label='Caution = 50')
    ax5.set_xlabel('Time (s)'); ax5.set_ylabel('cond(J)')
    ax5.set_title('Jacobian condition number')
    ax5.legend(fontsize=7); ax5.grid(alpha=0.3, which='both')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'task33_line_simulation.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'\nSaved: {out}')
    return fig


# =============================================================================
#  ENTRY POINTS
# =============================================================================

def main(args=None):
    """Entry point for  ros2 run <pkg> task33_line_velocity"""
    rclpy.init(args=args)
    node = LineVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    if '--ros' in sys.argv:
        if not ROS2_OK:
            print('ERROR: rclpy not available. Source your ROS2 workspace first.')
            sys.exit(1)
        main()
    else:
        print('Running SIMULATION  (pass --ros to run the ROS2 node)\n')
        data = simulate_line(n_laps=2)
        plot_simulation(data)
        plt.show()