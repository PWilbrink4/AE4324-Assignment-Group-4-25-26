import numpy as np

'''This file defines the functions for the jacobian
    Running will evaluate the Jacobian for one selected joint vector q
'''

def Jacobian(q_vector):
    theta_shoulder = q_vector[0]
    theta_upper = q_vector[1]
    theta_lower = q_vector[2]
    theta_wrist = q_vector[3]
    theta_gripper = q_vector[4]

    dxdshoulder = (0.11257*np.sin(theta_upper) + 0.0052*np.sin(theta_lower + theta_upper) - 0.028*np.cos(theta_upper) - 0.1349*np.cos(theta_lower + theta_upper) - 0.1351*np.cos(theta_lower + theta_upper + theta_wrist) - 0.0306)*np.cos(theta_shoulder)
    dxdupper = (0.028*np.sin(theta_upper) + 0.1349*np.sin(theta_lower + theta_upper) + 0.1351*np.sin(theta_lower + theta_upper + theta_wrist) + 0.11257*np.cos(theta_upper) + 0.0052*np.cos(theta_lower + theta_upper))*np.sin(theta_shoulder)
    dxdlower = (0.1349*np.sin(theta_lower + theta_upper) + 0.1351*np.sin(theta_lower + theta_upper + theta_wrist) + 0.0052*np.cos(theta_lower + theta_upper))*np.sin(theta_shoulder)
    dxdwrist = 0.1351*np.sin(theta_shoulder)*np.sin(theta_lower + theta_upper + theta_wrist)
    dxdgripper = 0

    dydshoulder = (0.11257*np.sin(theta_upper) + 0.0052*np.sin(theta_lower + theta_upper) - 0.028*np.cos(theta_upper) - 0.1349*np.cos(theta_lower + theta_upper) - 0.1351*np.cos(theta_lower + theta_upper + theta_wrist) - 0.0306)*np.sin(theta_shoulder)
    dydupper = -(0.028*np.sin(theta_upper) + 0.1349*np.sin(theta_lower + theta_upper) + 0.1351*np.sin(theta_lower + theta_upper + theta_wrist) + 0.11257*np.cos(theta_upper) + 0.0052*np.cos(theta_lower + theta_upper))*np.cos(theta_shoulder)
    dydlower = -(0.1349*np.sin(theta_lower + theta_upper) + 0.1351*np.sin(theta_lower + theta_upper + theta_wrist) + 0.0052*np.cos(theta_lower + theta_upper))*np.cos(theta_shoulder)
    dydwrist = -0.1351*np.sin(theta_lower + theta_upper + theta_wrist)*np.cos(theta_shoulder)
    dydgripper = 0

    dzdshoulder = 0
    dzdupper = -0.11257*np.sin(theta_upper) - 0.0052*np.sin(theta_lower + theta_upper) + 0.028*np.cos(theta_upper) + 0.1349*np.cos(theta_lower + theta_upper) + 0.1351*np.cos(theta_lower + theta_upper + theta_wrist)
    dzdlower = -0.0052*np.sin(theta_lower + theta_upper) + 0.1349*np.cos(theta_lower + theta_upper) + 0.1351*np.cos(theta_lower + theta_upper + theta_wrist)
    dzdwrist = 0.1351*np.cos(theta_lower + theta_upper + theta_wrist)
    dzdgripper = 0

    Jacobian = np.matrix([
        [dxdshoulder,dxdupper,dxdlower,dxdwrist,dxdgripper],
        [dydshoulder,dydupper,dydlower,dydwrist,dydgripper],
        [dzdshoulder,dzdupper,dzdlower,dzdwrist,dzdgripper],
        [0,1,1,1,0],
        [0,0,0,0,1]
    ])

    return Jacobian

#1 [-0.912, 0.645, -0.876, 0.232, 2.22] det(J) = 0.0032827987487133283
#2.1 [-1.303, -0.182, 0.446, 1.307, 0.0] det = 0.0024233546921401533
#2.2 [1.838, 1.265, 1.288, -0.982, 0.0] -1.3829512946227156e-06
#3 [0.0, 0.990, 1.288, -1.492, 0.001]
#4 None
#5 [-0.0,0.735,1.288,-1.238,3.141]

q = [-0.0,0.735,1.288,-1.238,3.141]
J_out = Jacobian(q)
print(np.linalg.det(J_out))
print(J_out)
