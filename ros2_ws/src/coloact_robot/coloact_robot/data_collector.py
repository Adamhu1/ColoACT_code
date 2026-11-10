#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import h5py
import time
import os

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # --- 配置存储路径 ---
        # 数据将保存在 ros2_ws 下的一个 dataset 文件夹中
        self.save_dir = os.path.expanduser("~/ros2_ws/src/coloact_robot/dataset")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # --- 状态标志 ---
        self.is_recording = False
        self.h5_file = None
        self.bridge = CvBridge()
        
        # --- 数据缓存 (用于同步) ---
        self.latest_depth = None
        self.latest_elev = None
        self.latest_cmd = None # [v, w]
        
        # --- 订阅话题 ---
        # 1. 订阅深度估计节点发布的图像
        self.create_subscription(Image, '/processed_depth', self.depth_cb, 10)
        self.create_subscription(Image, '/processed_elevation', self.elev_cb, 10)
        
        # 2. 订阅手柄控制器发布的指令 (这就是我们要模仿的"专家动作")
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        
        # 3. 订阅手柄按键 (用于控制录制开始/停止)
        self.create_subscription(Joy, 'joy', self.joy_cb, 10)
        
        # --- 定时器 ---
        # 以 10Hz 的频率将对齐的数据写入文件
        self.create_timer(0.1, self.record_step)
        
        self.get_logger().info(f"数据采集节点已就绪。保存路径: {self.save_dir}")
        self.get_logger().info("操作说明: 手柄 [START] 键开始录制，[BACK/SELECT] 键停止录制。")

    def joy_cb(self, msg):
        # 注意：不同手柄的按键映射可能不同
        # 这里假设 index 7 是 START, index 6 是 BACK/SELECT
        # 如果你的手柄不同，请通过 'ros2 topic echo /joy' 确认 index
        START_BUTTON_IDX = 7
        BACK_BUTTON_IDX = 6
        
        if len(msg.buttons) > START_BUTTON_IDX and msg.buttons[START_BUTTON_IDX] == 1:
            if not self.is_recording:
                self.start_recording()
        
        elif len(msg.buttons) > BACK_BUTTON_IDX and msg.buttons[BACK_BUTTON_IDX] == 1:
            if self.is_recording:
                self.stop_recording()

    def start_recording(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"demo_{timestamp}.h5")
        self.h5_file = h5py.File(filename, 'w')
        
        # 创建可变长度的数据集 (chunked storage)
        # 图像尺寸调整为 320x240 以节省空间和训练开销
        self.ds_depth = self.h5_file.create_dataset('depth', (0, 240, 320, 1), maxshape=(None, 240, 320, 1), dtype='f4')
        self.ds_elev = self.h5_file.create_dataset('elevation', (0, 240, 320, 1), maxshape=(None, 240, 320, 1), dtype='u1') # 伪高程通常是0-255灰度
        self.ds_action = self.h5_file.create_dataset('action', (0, 2), maxshape=(None, 2), dtype='f4') # v, w
        
        self.is_recording = True
        self.get_logger().info(f"🔴 开始录制: {filename}")

    def stop_recording(self):
        self.is_recording = False
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
        self.get_logger().info("⬛ 录制结束并保存。")

    def depth_cb(self, msg):
        try:
            # 现在使用mono16格式 (16位无符号整数)
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            # 转换回浮点数并归一化到[0,1]范围
            cv_img = cv_img.astype(np.float32) / 65535.0
            self.latest_depth = cv2.resize(cv_img, (320, 240))
        except Exception as e:
            self.get_logger().warn(f"深度图转换失败: {e}")
            # 尝试32FC1格式作为备用
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                self.latest_depth = cv2.resize(cv_img, (320, 240))
                self.get_logger().info("深度图转换成功 (使用32FC1格式)")
            except Exception as e2:
                self.get_logger().error(f"深度图转换完全失败: {e2}")
                self.latest_depth = None

    def elev_cb(self, msg):
        try:
            # 伪高程图通常是 mono8 (8位灰度)
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.latest_elev = cv2.resize(cv_img, (320, 240))
        except Exception as e:
            self.get_logger().warn(f"高程图转换失败: {e}")
            # 尝试其他编码方式
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                self.latest_elev = cv2.resize(cv_img, (320, 240))
                self.get_logger().info("高程图转换成功 (使用备用编码)")
            except Exception as e2:
                self.get_logger().error(f"高程图转换完全失败: {e2}")
                self.latest_elev = None

    def cmd_cb(self, msg):
        # 记录线速度 v 和 角速度 w
        self.latest_cmd = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    def record_step(self):
        # 只有在录制状态且所有数据都到位时才写入
        if not self.is_recording:
            return
        
        if self.latest_depth is None or self.latest_elev is None or self.latest_cmd is None:
            # 可以选择打印警告，或者静默等待数据流
            return

        # 写入 HDF5
        idx = self.ds_action.shape[0] # 获取当前行数
        
        # 扩展数据集大小
        self.ds_depth.resize(idx+1, axis=0)
        self.ds_elev.resize(idx+1, axis=0)
        self.ds_action.resize(idx+1, axis=0)
        
        # 填入数据 (增加通道维度以符合 (H, W, C) 格式)
        self.ds_depth[idx] = self.latest_depth[:, :, np.newaxis]
        self.ds_elev[idx] = self.latest_elev[:, :, np.newaxis]
        self.ds_action[idx] = self.latest_cmd
        
        # 打印进度
        print(f"\r正在录制... 帧数: {idx+1}", end="")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.is_recording:
            node.stop_recording()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()