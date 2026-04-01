The controller files folder contains the files used to perform the real-life tasks with the robot (in addition to being able to run them in simulation).
- Task 2: position_hold
- Task 2.3: trajectory_smooth
- Task 3: velocity_control
- Task 4: simV4 (pick and place)
- Task 4: simV5 (stacking)
 
Note:  Only the velocity_control.py file has an additional function, which is providing the plots for task 3.3 in the report. Furthermore, the robot configuration files are added as a reference, listing some of the different home positions used throughout the project

The files in the Kinematics folder were used for multiple purposes that do not directly interact with the robot. Mostly, these files were later integrated into the controllers. To use them it is important to keep them in the same folder as they reference each other, whilst this is not needed for the controller files. The different purposes are:

Define and test the various functions for the kinematics of the robot,
such as the inverse kinematics, forward kinematics, and Jacobian:

- DirectForwardKinematics.py
- InverseKinematics.py
- JacobianDef.py

Additionally, perform certain standalone tasks that didnt require the robot were performed, such as plotting the workspace and finding the symbolic expressions of
the different kinematics functions:

- RobotConstants.py (Defines joint limits)
- WorkspacePlotting.py
- SympyForwardKinematics.py

Furthermore, perform tasks in preparation of runs with the real life robot, such as testing and visualisation of the trajectory:

TrajectoryTest.py
WorkspacePlotting.py


