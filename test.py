import numpy as np
from utils.function import vector_to_axis_angles
from utils.function import angles_to_quaternion
from utils.function import quaternion_multiply
from utils.function import quatertion_conj
from utils.function import quaternion_norm


v = [3, 4, 5]
angles = vector_to_axis_angles(v)
angle_x = angles['angle_x (rad)']
angle_y = angles['angle_y (rad)']
angle_z = angles['angle_z (rad)']

angles = {
    'angle_x (rad)': angle_x,
    'angle_y (rad)': angle_y, 
    'angle_z (rad)': angle_z
}

quaternion1 = angles_to_quaternion(angles)
print(f"组合旋转的四元数为 [w, x, y, z]: {quaternion1}")


v = [1, 2, 3]
angles = vector_to_axis_angles(v)
angle_x = angles['angle_x (rad)']
angle_y = angles['angle_y (rad)']
angle_z = angles['angle_z (rad)']

angles = {
    'angle_x (rad)': angle_x,
    'angle_y (rad)': angle_y, 
    'angle_z (rad)': angle_z
}

quaternion2 = angles_to_quaternion(angles)
print(f"组合旋转的四元数为 [w, x, y, z]: {quaternion2}")


quaternion = quaternion_multiply(quatertion_conj(quaternion1), quaternion2)
print(f"四元数乘法的结果为 [w, x, y, z]: {quaternion}")

quaternion_norm = quaternion_norm(quaternion)
print(f"四元数归一化为: {quaternion_norm}")