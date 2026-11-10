#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import os
import time
from pathlib import Path
import sys
from threading import Thread, Lock
import queue

# ================= 配置区域 =================
# 请务必确认这里的路径是你安装了 torch 和 depth_anything_v2 的环境路径
CONDA_ENV_PATH = "/path/to/conda/env"
# ===========================================

def main_with_conda():
    """
    环境切换包装器：
    如果当前不是在指定的 conda 环境中运行，则自动切换环境并重启脚本。
    这对于在 ROS2 launch 系统中调用需要特定依赖（如 torch）的节点非常有用。
    """
    import subprocess
    
    python_executable = f"{CONDA_ENV_PATH}/bin/python3.8"

    # 检查目标解释器是否存在
    if not os.path.exists(python_executable):
        print(f"错误: 找不到 Python 解释器: {python_executable}")
        print(f"请检查 CONDA_ENV_PATH 配置是否正确: {CONDA_ENV_PATH}")
        return

    # 设置环境变量
    env = os.environ.copy()
    env["PATH"] = f"{CONDA_ENV_PATH}/bin:{env['PATH']}"
    # 注入 site-packages 路径以确保能找到库
    env["PYTHONPATH"] = f"{CONDA_ENV_PATH}/lib/python3.8/site-packages:{env.get('PYTHONPATH', '')}"

    print(f"正在切换到 Conda 环境: {CONDA_ENV_PATH} ...")
    
    # 重新运行当前脚本
    current_script = os.path.abspath(__file__)
    # 传递原本的命令行参数（这对 ROS2 节点很重要）
    subprocess.run([python_executable, current_script] + sys.argv[1:], env=env)

class DepthProcessingNode(Node):
    def __init__(self):
        super().__init__('depth_estimator') # 节点名称与 setup.py 入口保持逻辑一致

        # 初始化参数
        self.declare_parameter('camera_id', 0)
        # 修改默认尺寸为 320x240，与模仿学习数据采集尺寸一致，提升速度
        self.declare_parameter('processing_size', [320, 240]) 
        self.declare_parameter('processing_frequency', 15.0) # 稍微提高频率
        self.declare_parameter('debug_mode', False)

        # 获取参数
        camera_id = self.get_parameter('camera_id').value
        processing_size = tuple(self.get_parameter('processing_size').value)
        processing_frequency = self.get_parameter('processing_frequency').value
        self.debug_mode = self.get_parameter('debug_mode').value

        self.get_logger().info(f'初始化深度节点 - 相机: {camera_id}, 尺寸: {processing_size}')

        # --- 发布者配置 (与 Data Collector 对齐) ---
        
        # 1. 相对深度图 (32FC1) -> 输入到模仿学习网络通道1
        self.depth_pub = self.create_publisher(
            Image,
            '/processed_depth',
            10
        )

        # 2. 伪高程图 (mono8) -> 输入到模仿学习网络通道2
        self.elevation_pub = self.create_publisher(
            Image,
            '/processed_elevation',
            10
        )

        # 3. 彩色可视化 (bgr8) -> 仅供人眼调试，不录入数据集
        self.colored_elevation_pub = self.create_publisher(
            Image,
            '/colored_elevation',
            10
        )

        self.raw_pub = self.create_publisher(Image, '/camera/raw', 10)

        # 初始化处理核心
        self.depth_system = RealTimeDepthElevation(camera_id, processing_size, self)

        # 定时器
        self.timer = self.create_timer(
            1.0 / processing_frequency,
            self.process_depth
        )

        self.frame_count = 0
        self.get_logger().info('深度估计节点已启动')

    def process_depth(self):
        """主处理循环"""
        try:
            results = self.depth_system.get_latest_results()

            if results is not None:
                # 发布 ROS 消息
                # 时间戳统一使用当前时间，方便数据采集节点对齐
                now = self.get_clock().now().to_msg()
                
                depth_msg = self.create_depth_image_msg(results['relative_depth'], now)
                self.depth_pub.publish(depth_msg)

                elev_msg = self.create_elevation_image_msg(results['elevation_map'], now)
                self.elevation_pub.publish(elev_msg)

                color_msg = self.create_colored_elevation_msg(results['colored_elevation'], now)
                self.colored_elevation_pub.publish(color_msg)

                self.frame_count += 1
                if self.debug_mode and self.frame_count % 30 == 0:
                    self.get_logger().info(f'Publishing... Frame {self.frame_count}')
                # <--- 新增发布原图逻辑
                if 'raw_image' in results:
                    # 复用 create_colored_elevation_msg 因为格式也是 bgr8
                    raw_msg = self.create_colored_elevation_msg(results['raw_image'], now)
                    self.raw_pub.publish(raw_msg)

        except Exception as e:
            self.get_logger().error(f'处理异常: {str(e)}')

    def create_depth_image_msg(self, depth_data, timestamp):
        msg = Image()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_depth_frame'
        if not isinstance(depth_data, np.ndarray): depth_data = np.array(depth_data)
        msg.height = depth_data.shape[0]
        msg.width = depth_data.shape[1]

        # 使用mono16格式避免cv_bridge的GDAL库冲突
        # 将深度数据转换为16位无符号整数
        depth_uint16 = (depth_data * 65535.0).clip(0, 65535).astype(np.uint16)

        msg.encoding = 'mono16'
        msg.is_bigendian = 0
        msg.step = msg.width * 2
        msg.data = depth_uint16.tobytes()
        return msg

    def create_elevation_image_msg(self, elevation_data, timestamp):
        msg = Image()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_elevation_frame'
        if not isinstance(elevation_data, np.ndarray): elevation_data = np.array(elevation_data)
        msg.height = elevation_data.shape[0]
        msg.width = elevation_data.shape[1]
        msg.encoding = 'mono8'
        msg.is_bigendian = 0
        msg.step = msg.width
        msg.data = elevation_data.astype(np.uint8).tobytes()
        return msg

    def create_colored_elevation_msg(self, colored_data, timestamp):
        msg = Image()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_color_frame'
        if not isinstance(colored_data, np.ndarray): colored_data = np.array(colored_data)
        msg.height = colored_data.shape[0]
        msg.width = colored_data.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = colored_data.astype(np.uint8).tobytes()
        return msg

    def destroy_node(self):
        if hasattr(self, 'depth_system'):
            self.depth_system.stop()
        super().destroy_node()

class RealTimeDepthElevation:
    def __init__(self, camera_id=0, processing_size=(320, 240), ros_node=None):
        self.camera_id = camera_id
        self.processing_size = processing_size
        self.ros_node = ros_node
        self.running = False
        self.frame_lock = Lock()
        self.current_frame = None
        self.results_queue = queue.Queue(maxsize=1)

        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        # 尝试设置高分辨率采集，然后软件缩小，保证视野
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self._log_error(f"无法打开摄像头 {camera_id}")
            return

        # 导入模型 (假设环境已切换成功，这里直接导入)
        self._init_model()

    def _log_info(self, msg):
        if self.ros_node: self.ros_node.get_logger().info(msg)
        else: print(msg)

    def _log_error(self, msg):
        if self.ros_node: self.ros_node.get_logger().error(msg)
        else: print(msg)

    def _init_model(self):
        # 这里假定 Depth-Anything-V2 代码库在用户主目录下
        # 如果你的路径不同，请修改这里
        depth_anything_path = Path.home() / "depth_anything_v2" / "Depth-Anything-V2"
        if str(depth_anything_path) not in sys.path:
            sys.path.append(str(depth_anything_path))

        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._log_info(f"使用计算设备: {self.device}")

            # 配置模型
            self.model = DepthAnythingV2(
                encoder='vitb', 
                features=128, 
                out_channels=[96, 192, 384, 768]
            )
            
            # 加载权重
            checkpoint_path = Path.home() / "checkpoints" / "depth_anything_v2_vitb.pth"
            if not checkpoint_path.exists():
                self._log_error(f"权重文件不存在: {checkpoint_path}")
                raise FileNotFoundError(f"Check {checkpoint_path}")

            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            
            # 兼容不同的 checkpoint 格式
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model = self.model.to(self.device).eval()
            self._log_info("Depth Anything V2 模型加载完毕")

        except Exception as e:
            self._log_error(f"模型初始化失败: {e}")
            self.model = None

    def start_capture(self):
        self.running = True
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.process_thread = Thread(target=self._process_frames, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
            else:
                time.sleep(0.1)

    def _process_frames(self):
        while self.running:
            if self.current_frame is None or self.model is None:
                time.sleep(0.01)
                continue

            try:
                with self.frame_lock:
                    frame_input = self.current_frame.copy()
                
                # ================= 修改开始 =================
                # 1. ❌ 删除原来的先缩小逻辑
                # frame_resized = cv2.resize(frame_input, self.processing_size) 
                
                # 2. ✅ 使用高清原图 (640x480) 进行推理
                # Depth Anything V2 在高分图上能捕捉到更细微的深度变化
                raw_depth_large = self.infer_depth(frame_input)
                
                # 3. 后处理生成高程图
                # 归一化深度 [0, 1]
                norm_depth_large = (raw_depth_large - raw_depth_large.min()) / (raw_depth_large.max() - raw_depth_large.min() + 1e-8)
                
                # ✅ 关键步骤：在高清图上计算梯度高程
                # 这样微小的息肉也能产生强烈的梯度响应（高亮像素）
                elevation_large = self.compute_pseudo_elevation(norm_depth_large)
                
                # 4. ✅ 降采样 (Downsample) 到目标尺寸 (320x240)
                # 使用 INTER_AREA 插值，它是进行图像缩小的最佳算法，能保留“亮斑”特征
                norm_depth_small = cv2.resize(norm_depth_large, self.processing_size, interpolation=cv2.INTER_AREA)
                elevation_small = cv2.resize(elevation_large, self.processing_size, interpolation=cv2.INTER_AREA)
                
                # 生成可视化图 (用缩小后的，节省显示资源)
                colored_elev = cv2.applyColorMap(elevation_small, cv2.COLORMAP_JET)

                # 5. 存入队列
                result = {
                    'raw_image': frame_input,      # 原图 (640x480) 供 RGB 使用
                    'relative_depth': norm_depth_small, # 发送给 Policy 的是 320x240
                    'elevation_map': elevation_small,   # 发送给 Policy 的是 320x240
                    'colored_elevation': colored_elev
                }
                # ================= 修改结束 =================
                
                if self.results_queue.full():
                    try: self.results_queue.get_nowait()
                    except: pass
                self.results_queue.put(result)
                
                # 简单的帧率控制
                time.sleep(0.01)

            except Exception as e:
                self._log_error(f"帧处理出错: {e}")
                time.sleep(0.1)

            except Exception as e:
                self._log_error(f"帧处理出错: {e}")
                time.sleep(0.1)

    def infer_depth(self, image):
        """调用模型进行推理"""
        # 准备输入: BGR -> RGB, Normalize, Tensor
        import torch
        import torch.nn.functional as F
        
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # 归一化参数 (ImageNet stats)
        image_rgb = (image_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # 模型推理要求输入尺寸是 14 的倍数
        target_h, target_w = h, w
        if h % 14 != 0: target_h = (h // 14) * 14
        if w % 14 != 0: target_w = (w // 14) * 14
        
        if target_h != h or target_w != w:
            image_tensor = F.interpolate(image_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

        with torch.no_grad():
            depth = self.model(image_tensor)
            
        # 恢复原始尺寸
        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
            
        return depth.cpu().numpy().squeeze()

    def compute_pseudo_elevation(self, normalized_depth):
        """基于深度的梯度计算伪高程图，突出皱襞结构"""
        depth_u8 = (normalized_depth * 255).astype(np.uint8)
        
        # Scharr 算子对边缘更敏感
        grad_x = cv2.Scharr(depth_u8, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(depth_u8, cv2.CV_64F, 0, 1)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 平滑和增强
        magnitude = cv2.GaussianBlur(magnitude, (3, 3), 0)
        magnitude = np.log1p(magnitude) # Log 缩放增强细节
        
        # 归一化回 0-255
        elevation = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        return (elevation * 255).astype(np.uint8)

    def get_latest_results(self):
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    """ROS2 节点入口点"""
    # 检查 Torch 是否可用，不可用则切换环境
    try:
        import torch
    except ImportError:
        print("当前环境缺少 torch，尝试切换到 Conda 环境...")
        main_with_conda()
        return

    # 初始化 ROS2
    rclpy.init(args=args)

    node = None
    try:
        node = DepthProcessingNode()
        node.depth_system.start_capture()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"节点运行出错: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()