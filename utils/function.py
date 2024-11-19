import numpy as np

def vector_to_axis_angles(v):
    """
    计算三维向量 v = (x, y, z) 到 x、y、z 轴的夹角（以弧度表示）。
    参数：
        v: list or np.array，三维向量 [x, y, z]
    返回：
        angles: dict，包含与 x、y、z 轴的夹角
    """
    # 确保 v 是 NumPy 数组
    v = np.array(v)
    
    # 计算向量的模长
    norm = np.linalg.norm(v)
    
    if norm == 0:
        raise ValueError("zero!")
    
    # 计算与每个轴的夹角（弧度）
    angle_x = np.arccos(v[0] / norm)  # 与 x 轴夹角
    angle_y = np.arccos(v[1] / norm)  # 与 y 轴夹角
    angle_z = np.arccos(v[2] / norm)  # 与 z 轴夹角
    
    # 返回结果
    return {
        'angle_x (rad)': angle_x,
        'angle_y (rad)': angle_y,
        'angle_z (rad)': angle_z,
        'angle_x (deg)': np.degrees(angle_x),
        'angle_y (deg)': np.degrees(angle_y),
        'angle_z (deg)': np.degrees(angle_z),
    }





def angles_to_quaternion(angles):
    """
    将向量与三轴的夹角（弧度）转换为旋转四元数。
    参数：
        angles: dict，包含与 x、y、z 轴夹角的弧度值，键为 'angle_x', 'angle_y', 'angle_z'
    返回：
        quaternion: np.array，组合旋转的最终四元数 [w, x, y, z]
    """
    # 提取夹角
    theta_x = angles['angle_x (rad)']
    theta_y = angles['angle_y (rad)']
    theta_z = angles['angle_z (rad)']
    
    # 绕 x 轴的旋转四元数
    q_x = np.array([
        np.cos(theta_x / 2),
        np.sin(theta_x / 2),
        0,
        0
    ])
    
    # 绕 y 轴的旋转四元数
    q_y = np.array([
        np.cos(theta_y / 2),
        0,
        np.sin(theta_y / 2),
        0
    ])
    
    # 绕 z 轴的旋转四元数
    q_z = np.array([
        np.cos(theta_z / 2),
        0,
        0,
        np.sin(theta_z / 2)
    ])
    
    # 四元数乘法函数
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
    
    # 组合旋转：q_final = q_z * q_y * q_x
    q_final = quaternion_multiply(quaternion_multiply(q_z, q_y), q_x)
    
    return q_final



def quatertion_conj(quatertion):
    """
    function：求四元数的共轭
    :param quatertion:
    :return: 行向量 或者 n*4矩阵
    """
    quatertion = np.array(quatertion)
    # 判断是否为一个行向量
    if len(quatertion.shape)<2:
        quatertion = quatertion.reshape(1,-1)  # 转为行向量

    new_quat = np.zeros(quatertion.shape)
    for i in range(quatertion.shape[0]):
        new_quat[i, :] = np.array([1, -1, -1, -1]) * quatertion[i, :]
    return new_quat

def quaternion_norm(quaternion):
    """
    function : quaternion 的每一行归一化四元数
    :param quaternion:
    :return:
    """
    N = quaternion.shape[0]
    new_quaternion = np.zeros((quaternion.shape))
    for i in range(N):
        new_quaternion[i, :] = quaternion[i, :] / np.linalg.norm(quaternion[i, :])
    return new_quaternion

def quaternion_multiply(quaternion1, quaternion0):
    """
    function：计算四元数乘法
    :param quaternion1: n*4  若只有一行，必须是行向量。 经过修改后， 输入可以是一个任意的数组
    :param quaternion0: n*4  若只有一行，必须是行向量。 经过修改后， 输入可以是一个任意的数组
    :param dim:
    :return: quatertion1 * quaternion2
    """
    # quat1 * quat2
    quaternion0 = np.array(quaternion0)
    quaternion1 = np.array(quaternion1)
    if len(quaternion0.shape) < 2:
        quaternion0 = quaternion0.reshape(1, -1)
    if len(quaternion1.shape) < 2:
        quaternion1 = quaternion1.reshape(1, -1)

    max_length = max(quaternion0.shape[0], quaternion1.shape[0])   # 寻找最大行数
    result = np.zeros((max_length, 4))   # 输出结果
    if quaternion1.shape[0] == 1:
        quaternion1 = quaternion1 * np.ones((max_length, 4))
    if quaternion0.shape[0] == 1:
        quaternion0 = quaternion0 * np.ones((max_length, 4))
    for i in range(max_length):
        w0, x0, y0, z0 = quaternion0[i, :]
        w1, x1, y1, z1 = quaternion1[i, :]
        result[i, :] = np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,

                                 x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,

                                 -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,

                                 x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    return result