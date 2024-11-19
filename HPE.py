import cv2
import numpy as np
import mediapipe as mp
import socket
import json  # 用于将数据转换为JSON格式进行发送

from utils.function import vector_to_axis_angles, angles_to_quaternion, quaternion_multiply, quatertion_conj, quaternion_norm

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle = np.degrees(angle)
    return angle

# 初始化UDP通信
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)  # Unity端的IP和端口

# 初始化 MediaPipe 手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 左右相机的内参矩阵
mtx_l = np.array([[687.32247929, 0., 371.73986906],
                  [0., 689.73139961, 219.77006416],
                  [0., 0., 1.]])
mtx_r = np.array([[437.06184307, 0., 367.14042088],
                  [0., 443.91578956, 203.58730918],
                  [0., 0., 1.]])

# 旋转矩阵和平移向量
R = np.array([[0.90237799, -0.14775324, 0.40482458],
              [0.05073797, 0.99280611, 0.24066695],
              [-0.42797473, -0.19638186, 0.88215445]])
T = np.array([[-434.36001552],
              [-86.22432684],
              [-100.42173922]])

# 初始化摄像头
cap_left = cv2.VideoCapture(0)  # 左摄像头
cap_right = cv2.VideoCapture(1)  # 右摄像头

# 函数：三角测量计算3D手部关键点
def triangulate_hand_keypoints(corners_l, corners_r):
    points4D_homogeneous = cv2.triangulatePoints(
        mtx_l @ np.hstack((np.eye(3), np.zeros((3, 1)))),
        mtx_r @ np.hstack((R, T)),
        corners_l.T,
        corners_r.T
    )
    points4D = points4D_homogeneous / points4D_homogeneous[3]
    return points4D[:3].T

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("无法读取摄像头图像")
        break

    frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

    result_left = hands.process(frame_left_rgb)
    result_right = hands.process(frame_right_rgb)

    if result_left.multi_hand_landmarks and result_right.multi_hand_landmarks:
        hand_landmarks_left = result_left.multi_hand_landmarks[0]
        hand_landmarks_right = result_right.multi_hand_landmarks[0]

        keypoints_left = []
        keypoints_right = []
        for i in range(21):
            x_l = int(hand_landmarks_left.landmark[i].x * frame_left.shape[1])
            y_l = int(hand_landmarks_left.landmark[i].y * frame_left.shape[0])
            keypoints_left.append([x_l, y_l])

            x_r = int(hand_landmarks_right.landmark[i].x * frame_right.shape[1])
            y_r = int(hand_landmarks_right.landmark[i].y * frame_right.shape[0])
            keypoints_right.append([x_r, y_r])

        keypoints_left = np.array(keypoints_left, dtype=np.float32)
        keypoints_right = np.array(keypoints_right, dtype=np.float32)

        keypoints_3d = triangulate_hand_keypoints(keypoints_left, keypoints_right)

        # 创建一个字典来存储所有关键点的坐标和弯曲角度
        data_to_send = {
            "keypoints": [],  # 存储关键点坐标
            "angles": []  # 存储各手指关节的弯曲角度
        }

        # 打印并存储21个关键点的三维坐标
        for i, point in enumerate(keypoints_3d):
            # 打印每个关键点的三维坐标
            print(f"关键点 {i}: x={float(point[0])}, y={float(point[1])}, z={float(point[2])}")
            data_to_send["keypoints"].append({
                "index": i,
                "x": float(point[0]),  # 将numpy类型转换为Python的float类型
                "y": float(point[1]),
                "z": float(point[2])
            })

        # 计算所有指定的关节角度
        finger_joints = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),   # 大拇指关节点
            (0, 5, 6), (5, 6, 7), (6, 7, 8),   # 食指关节点
            (0, 9, 10), (9, 10, 11), (10, 11, 12),  # 中指关节点
            (0, 13, 14), (13, 14, 15), (14, 15, 16),  # 无名指关节点
            (0, 17, 18), (17, 18, 19), (18, 19, 20)  # 小拇指关节点
        ]
        for (a, b, c) in finger_joints:
            v1 = keypoints_3d[a] - keypoints_3d[b]
            v2 = keypoints_3d[b] - keypoints_3d[c]

            ###############################modified by ljk
            angles_v1 = vector_to_axis_angles(v1)
            angles_v2 = vector_to_axis_angles(v2)
            quaternion_v1 = angles_to_quaternion(angles_v1)
            quaternion_v2 = angles_to_quaternion(angles_v2)
            quaternion_v1_v2 = quaternion_multiply(quatertion_conj(quaternion_v1), quaternion_v2)
            quaternion_norm_v1_v2 = quaternion_norm(quaternion_v1_v2)
            print(quaternion_norm_v1_v2)
            ###############################modified by ljk

            angle = calculate_angle(v1, v2)
            # 打印关节角度
            print(f"关节 {b} 的弯曲角度: {float(angle)} 度")
            data_to_send["angles"].append({
                "joint": b,
                "angle": float(angle)  # 将numpy类型转换为Python的float类型
            })

        # 将数据转换为JSON字符串
        json_data = json.dumps(data_to_send)

        # 通过UDP发送数据到Unity
        sock.sendto(json_data.encode(), serverAddressPort)

        # 可视化手部关键点
        mp_drawing.draw_landmarks(frame_left, hand_landmarks_left, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_right, hand_landmarks_right, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Left Camera', frame_left)
    cv2.imshow('Right Camera', frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
