#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import h5py
import time
import os
import struct

#BC的数据收集用


class DataCollectorDirect(Node):
    def __init__(self):
        super().__init__('data_collector')

        # --- 配置存储路径 ---
        # 数据将保存在 ros2_ws 下的一个 dataset 文件夹中
        self.save_dir = os.path.expanduser("~/ros2_ws/src/coloact_robot/dataset")
        os.makedirs(self.save_dir, exist_ok=True)

        # --- 状态标志 ---
        self.is_recording = False
        self.h5_file = None

        # --- 数据缓存 (用于同步) ---
        self.latest_depth = None
        self.latest_elev = None
        self.latest_cmd = None # [v, w]
        self.latest_raw = None  # <--- 新增缓存变量
        # <--- 新增：用于缓存彩色高程图
        self.latest_colored_elev = None

        # --- 订阅话题 ---
        # 1. 订阅深度估计节点发布的图像
        self.create_subscription(Image, '/camera/raw', self.raw_cb, 10) # <--- 新增订阅
        self.create_subscription(Image, '/processed_depth', self.depth_cb, 10)
        self.create_subscription(Image, '/processed_elevation', self.elev_cb, 10)
        self.create_subscription(Image, '/colored_elevation', self.colored_elev_cb, 10)
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
            # 直接从ROS消息解析数据，避免cv_bridge
            if msg.encoding == 'mono16':
                # 16位灰度图像
                depth_data = np.frombuffer(msg.data, dtype=np.uint16)
                depth_data = depth_data.reshape((msg.height, msg.width))
                # 转换为浮点数并归一化
                depth_data = depth_data.astype(np.float32) / 65535.0
                self.latest_depth = depth_data
            elif msg.encoding == '32FC1':
                # 32位浮点深度图
                depth_data = np.frombuffer(msg.data, dtype=np.float32)
                depth_data = depth_data.reshape((msg.height, msg.width))
                self.latest_depth = depth_data
            else:
                self.get_logger().warn(f"不支持的深度图格式: {msg.encoding}")
                self.latest_depth = None

        except Exception as e:
            self.get_logger().error(f"深度图解析失败: {e}")
            self.latest_depth = None

    def colored_elev_cb(self, msg):
        """解析彩色高程可视化图"""
        try:
            if msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape((msg.height, msg.width, 3))
                self.latest_colored_elev = img
        except Exception as e:
            self.get_logger().error(f"彩色高程图解析失败: {e}")

    def elev_cb(self, msg):
        try:
            # 直接从ROS消息解析数据，避免cv_bridge
            if msg.encoding == 'mono8':
                # 8位灰度图像
                elev_data = np.frombuffer(msg.data, dtype=np.uint8)
                elev_data = elev_data.reshape((msg.height, msg.width))
                self.latest_elev = elev_data
            elif msg.encoding == 'bgr8':
                # BGR彩色图像，转换为灰度
                color_data = np.frombuffer(msg.data, dtype=np.uint8)
                color_data = color_data.reshape((msg.height, msg.width, 3))
                # 简单的BGR到灰度转换
                elev_data = 0.299 * color_data[:,:,2] + 0.587 * color_data[:,:,1] + 0.114 * color_data[:,:,0]
                elev_data = elev_data.astype(np.uint8)
                self.latest_elev = elev_data
            else:
                self.get_logger().warn(f"不支持的高程图格式: {msg.encoding}")
                self.latest_elev = None

        except Exception as e:
            self.get_logger().error(f"高程图解析失败: {e}")
            self.latest_elev = None

    def raw_cb(self, msg):
            """手动解析 BGR8 原图"""
            try:
                if msg.encoding == 'bgr8':
                    img = np.frombuffer(msg.data, dtype=np.uint8)
                    img = img.reshape((msg.height, msg.width, 3))
                    self.latest_raw = img
            except Exception as e:
                self.get_logger().error(f"原图解析失败: {e}")

    def cmd_cb(self, msg):
        # 记录线速度 v 和 角速度 w
        self.latest_cmd = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    def record_step(self):
        # ==================== 2x2 实时显示网格 ====================
        # 定义统一的显示尺寸 (宽, 高) - 建议与录制尺寸一致
        W, H = 320, 240
        
        # --- 1. 左上角: 原始图像 (Original) ---
        if self.latest_raw is not None:
            # 缩放并拷贝，防止修改原数据
            img_raw = cv2.resize(self.latest_raw, (W, H))
        else:
            # 如果没数据，显示黑色背景
            img_raw = np.zeros((H, W, 3), dtype=np.uint8)
        # 添加文字标签
        cv2.putText(img_raw, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- 2. 右上角: 深度图 (Depth Map) ---
        if self.latest_depth is not None:
            # float32 (0-1) -> uint8 (0-255)
            depth_u8 = (self.latest_depth * 255).astype(np.uint8)
            # 应用伪彩色 (MAGMA 或 JET)
            img_depth = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
            img_depth = cv2.resize(img_depth, (W, H))
        else:
            img_depth = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(img_depth, "Depth Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- 3. 左下角: 高程图灰度 (Elevation Gray) ---
        if self.latest_elev is not None:
            img_elev_gray = cv2.resize(self.latest_elev, (W, H))
            # 关键：单通道转三通道，否则无法与彩色图拼接
            img_elev_gray = cv2.cvtColor(img_elev_gray, cv2.COLOR_GRAY2BGR)
        else:
            img_elev_gray = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(img_elev_gray, "Elevation (Gray)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- 4. 右下角: 高程图彩色 (Elevation Color) ---
        if self.latest_colored_elev is not None:
            img_elev_color = cv2.resize(self.latest_colored_elev, (W, H))
        else:
            img_elev_color = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(img_elev_color, "Elevation (Color)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- 拼接网格 ---
        # 顶部两张水平拼接
        top_row = np.hstack((img_raw, img_depth))
        # 底部两张水平拼接
        bottom_row = np.hstack((img_elev_gray, img_elev_color))
        # 上下垂直拼接
        grid = np.vstack((top_row, bottom_row))

        # --- 显示 ---
        cv2.imshow("Data Collection Monitor", grid)
        cv2.waitKey(1)





        # 只有在录制状态且所有数据都到位时才写入
        if not self.is_recording:
            return

        if self.latest_depth is None or self.latest_elev is None or self.latest_cmd is None:
            # 可以选择打印警告，或者静默等待数据流
            return

        try:
            # 写入 HDF5
            idx = self.ds_action.shape[0] # 获取当前行数

            # 扩展数据集大小
            self.ds_depth.resize(idx+1, axis=0)
            self.ds_elev.resize(idx+1, axis=0)
            self.ds_action.resize(idx+1, axis=0)

            # 确保图像尺寸为320x240
            if self.latest_depth.shape != (240, 320):
                depth_resized = np.zeros((240, 320), dtype=np.float32)
                # 简单的中心裁剪或缩放
                h, w = self.latest_depth.shape
                if h >= 240 and w >= 320:
                    # 中心裁剪
                    start_h = (h - 240) // 2
                    start_w = (w - 320) // 2
                    depth_resized = self.latest_depth[start_h:start_h+240, start_w:start_w+320]
                else:
                    # 缩放 (简单的最近邻)
                    scale_h = 240 / h
                    scale_w = 320 / w
                    for i in range(240):
                        for j in range(320):
                            orig_i = min(int(i / scale_h), h-1)
                            orig_j = min(int(j / scale_w), w-1)
                            depth_resized[i, j] = self.latest_depth[orig_i, orig_j]
                self.latest_depth = depth_resized

            if self.latest_elev.shape != (240, 320):
                elev_resized = np.zeros((240, 320), dtype=np.uint8)
                h, w = self.latest_elev.shape
                if h >= 240 and w >= 320:
                    # 中心裁剪
                    start_h = (h - 240) // 2
                    start_w = (w - 320) // 2
                    elev_resized = self.latest_elev[start_h:start_h+240, start_w:start_w+320]
                else:
                    # 缩放 (简单的最近邻)
                    scale_h = 240 / h
                    scale_w = 320 / w
                    for i in range(240):
                        for j in range(320):
                            orig_i = min(int(i / scale_h), h-1)
                            orig_j = min(int(j / scale_w), w-1)
                            elev_resized[i, j] = self.latest_elev[orig_i, orig_j]
                self.latest_elev = elev_resized

            # 填入数据 (增加通道维度以符合 (H, W, C) 格式)
            self.ds_depth[idx] = self.latest_depth[:, :, np.newaxis]
            self.ds_elev[idx] = self.latest_elev[:, :, np.newaxis]
            self.ds_action[idx] = self.latest_cmd

            # 打印进度
            print(f"\r正在录制... 帧数: {idx+1}", end="")

        except Exception as e:
            self.get_logger().error(f"数据写入失败: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorDirect()
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