from InverseKinematics import *

'''Running this file will generate an animation style python visualisation of the defined trajectory in TrajectoryTest.py'''


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
    TJ0, TJ1, TJ2, TJ3, TJ4, TJ5, TJ6 = ForwardKinematicsFullOutput_PostRotationFrame(q_history[i])

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
    ax.scatter(cartesian_shape[:, 0], cartesian_shape[:, 1], cartesian_shape[:, 2], label="Waypoint",color="orange")

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