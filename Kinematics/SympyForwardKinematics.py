import numpy as np
import sympy as sp


theta_shoulder = sp.symbols('theta_shoulder') #Shoulder joint
theta_upper = sp.symbols('theta_upper') #Upper arm joint
theta_lower = sp.symbols('theta_lower') #Lower arm  joint
theta_wrist = sp.symbols('theta_wrist') #Wrist joint
theta_gripper = sp.symbols('theta_gripper') #Gripper roll joint

def SympyRotationMatrix_X(sympy_angle):
    Rx = sp.Matrix([
        [1,0,0],
        [0,sp.cos(sympy_angle),-sp.sin(sympy_angle)],
        [0,sp.sin(sympy_angle),sp.cos(sympy_angle)]
    ])
    return Rx

def SympyRotationMatrix_Y(sympy_angle):
    Ry = sp.Matrix([
        [sp.cos(sympy_angle), 0, sp.sin(sympy_angle)],
        [0, 1, 0],
        [-sp.sin(sympy_angle), 0, sp.cos(sympy_angle)],
    ])
    return Ry

def SympyRotationMatrix_Z(sympy_angle):
    Rz = sp.Matrix([
        [sp.cos(sympy_angle), -sp.sin(sympy_angle), 0],
        [sp.sin(sympy_angle), sp.cos(sympy_angle), 0],
        [0, 0, 1]
    ])
    return Rz

def SympyTranslationMatrix(x,y,z):
    P = sp.Matrix([
        [x],
        [y],
        [z]
    ])
    return P

def SympyHomogeneousTransformation(rotation_matrix,translation_matrix):
    T = rotation_matrix.copy()
    T = T.col_insert(3,translation_matrix)
    T = T.row_insert(3,sp.Matrix([[0,0,0,1]]))
    return T

Zero_translation = SympyTranslationMatrix(0,0,0)
#R_i_j means rotation of frame j with respect to frame i
R_world_base = SympyRotationMatrix_Z(sp.pi)
P_world_base = SympyTranslationMatrix(0, 0, 0)
T_world_base = SympyHomogeneousTransformation(R_world_base, P_world_base)

R_base_shoulder = SympyRotationMatrix_X(0)
P_base_shoulder = SympyTranslationMatrix(0,-0.0452,0.0165)
T_base_shoulder = SympyHomogeneousTransformation(R_base_shoulder,P_base_shoulder)

R_shoulder_upper = SympyRotationMatrix_Y(-sp.pi/2)
P_shoulder_upper = SympyTranslationMatrix(0, -0.0306,0.1025)
T_shoulder_upper = SympyHomogeneousTransformation(SympyRotationMatrix_Z(theta_shoulder),Zero_translation)*SympyHomogeneousTransformation(R_shoulder_upper,P_shoulder_upper)

R_upper_lower = SympyRotationMatrix_Z(0)
P_upper_lower = SympyTranslationMatrix(0.11257, -0.028, 0)
T_upper_lower = SympyHomogeneousTransformation(SympyRotationMatrix_Z(theta_upper),Zero_translation)*SympyHomogeneousTransformation(R_upper_lower,P_upper_lower)

#sp.pprint(T_world_base*T_base_shoulder*T_shoulder_upper)

R_lower_wrist = SympyRotationMatrix_Z(sp.pi/2)
P_lower_wrist = SympyTranslationMatrix(0.0052,-0.1349,0)
T_lower_wrist = SympyHomogeneousTransformation(SympyRotationMatrix_Z(theta_lower),Zero_translation)*SympyHomogeneousTransformation(R_lower_wrist,P_lower_wrist)

R_wrist_gripper = SympyRotationMatrix_Y(-sp.pi/2)
P_wrist_gripper = SympyTranslationMatrix(-0.0601,0,0)
T_wrist_gripper = SympyHomogeneousTransformation(SympyRotationMatrix_Z(theta_wrist),Zero_translation)*SympyHomogeneousTransformation(R_wrist_gripper,P_wrist_gripper)

R_gripper_grippercenter = SympyRotationMatrix_Z(0)
P_gripper_grippercenter = SympyTranslationMatrix(0,0,0.075)
T_gripper_grippercenter = SympyHomogeneousTransformation(SympyRotationMatrix_Z(theta_gripper),Zero_translation)*SympyHomogeneousTransformation(R_gripper_grippercenter,P_gripper_grippercenter)

T_world_grippercenter = T_world_base*T_base_shoulder*T_shoulder_upper*T_upper_lower*T_lower_wrist*T_wrist_gripper*T_gripper_grippercenter

sp.simplify(T_world_grippercenter)

sp.pprint(T_world_grippercenter[2,2])

