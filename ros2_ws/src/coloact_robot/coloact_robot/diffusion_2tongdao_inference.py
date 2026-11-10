#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
import torch
import numpy as np
import message_filters
from collections import deque
import threading
import time

# 引入你的 Diffusion 模型
# 确保 model_diffusion.py 在同一目录或 PYTHONPATH 中
try:
    from coloact_robot.model_diffusion import ColonDiffusionPolicy
except ImportError:
    from model_diffusion import ColonDiffusionPolicy

class DiffusionInferenceNode(Node):
    def __init__(self):
        super().__init__('diffusion_inference_node')

        # === 1. 配置参数 ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "/path/to/ros2_ws/src/coloact_robot/coloact_robot/diffusion_ckpt_80.pth"
        
        self.pred_horizon = 16  # 模型一次预测多少步
        self.action_horizon = 8 # 我们实际执行多少步 (执行一半就重新规划，保证反应灵敏)
        self.control_rate = 15.0 # 控制频率 Hz (电机执行动作的频率)
        
        # === 2. 加载模型 ===
        self.get_logger().info(f"正在加载 Diffusion 模型: {self.model_path}")
        # self.model = ColonDiffusionPolicy(pred_horizon=self.pred_horizon).to(self.device)
        self.model = ColonDiffusionPolicy(pred_horizon=self.pred_horizon, num_skills=4).to(self.device)


        if self.device.type == 'cpu':
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(self.model_path))
            
        self.model.eval()
        self.get_logger().info("模型加载完毕！")

        # === 3. 状态与队列 ===
        self.action_queue = deque(maxlen=self.pred_horizon) # 动作队列
        self.latest_obs = None # 最新的观测数据 (image tensor)
        self.latest_skill = torch.tensor([0], device=self.device).long() # 默认为直行
        self.lock = threading.Lock()
        
        # === 4. 通信接口 ===
        # 订阅 Skill ID (来自 VLM 或手柄)
        self.create_subscription(Int8, '/skill_id', self.skill_callback, 10)
        
        # 订阅图像 (Depth + Elevation)
        self.depth_sub = message_filters.Subscriber(self, Image, '/processed_depth')
        self.elev_sub = message_filters.Subscriber(self, Image, '/processed_elevation')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.elev_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.obs_callback)
        
        # 发布控制指令
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # === 5. 启动执行定时器 (Execution Loop) ===
        # 这是一个独立的节拍器，不管推理快慢，它负责按固定频率把动作发给电机
        self.create_timer(1.0 / self.control_rate, self.execution_loop)

        self.get_logger().info("Diffusion 推理节点已启动！等待图像...")

    def skill_callback(self, msg):
        """更新当前的原子任务 ID"""
        # 限制范围 0-3
        skill_id = max(0, min(3, msg.data))
        self.latest_skill = torch.tensor([skill_id], device=self.device).long()

    def obs_callback(self, depth_msg, elev_msg):
        """
        图像回调：
        1. 处理图像为 Tensor
        2. 检查队列是否快空了
        3. 如果快空了，触发推理 (Replanning)
        """
        try:
            # 数据预处理 (与 dataset_diffusion.py 保持一致)
            depth_data = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((240, 320))
            depth_f32 = depth_data.astype(np.float32) / 65535.0
            
            elev_data = np.frombuffer(elev_msg.data, dtype=np.uint8).reshape((240, 320))
            elev_f32 = elev_data.astype(np.float32) / 255.0
            
            # Stack -> [2, 240, 320]
            img_stack = np.stack([depth_f32, elev_f32], axis=0)
            
            # 转 Tensor -> [1, 2, 240, 320]
            with self.lock:
                self.latest_obs = torch.from_numpy(img_stack).unsqueeze(0).float().to(self.device)

            # === 关键逻辑：滑动窗口推理 ===
            # 如果队列里的动作少于阈值（比如只剩 2 步了），立刻重新规划
            # 这样既保证了动作的连续性，又利用了最新的图像
            if len(self.action_queue) <= 2:
                self.run_inference()

        except Exception as e:
            self.get_logger().error(f"图像处理出错: {e}")

    def run_inference(self):
        """执行 Diffusion 推理"""
        if self.latest_obs is None:
            return

        start_time = time.time()
        
        with torch.no_grad():
            # 1. 预测未来 16 步动作轨迹 [1, 16, 2]
            # predict_action 是我们在 model_diffusion.py 里写的去噪循环
            action_traj = self.model.predict_action(self.latest_obs, self.latest_skill)
            
            # 转 numpy
            action_traj = action_traj.cpu().numpy()[0] # [16, 2]

        # 2. 动作截断 (Action Horizon)
        # 我们只取前 8 步。因为 8 步之后的世界太遥远，可能不准，
        # 等执行完这 8 步，新的图像来了，我们会再次预测，覆盖后面的计划。
        new_actions = action_traj[:self.action_horizon]

        # 3. 填入队列
        with self.lock:
            # 清空旧的剩余动作（Receding Horizon 策略通常会覆盖旧计划）
            self.action_queue.clear() 
            for act in new_actions:
                self.action_queue.append(act)
        
        # debug: 打印推理耗时
        # print(f"Inference time: {(time.time() - start_time)*1000:.1f} ms")

    def execution_loop(self):
        """
        执行循环：以固定频率从队列取动作发送
        """
        cmd = Twist()
        
        should_publish = False
        
        with self.lock:
            if len(self.action_queue) > 0:
                # 取出队首动作
                action = self.action_queue.popleft() # [v, w]
                
                # 简单的限幅保护
                cmd.linear.x = float(np.clip(action[0], -0.2, 0.2))
                cmd.angular.z = float(np.clip(action[1], -1.0, 1.0))
                should_publish = True
            else:
                # 队列空了（推理没跟上，或者没图像），停车保护
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                should_publish = True
        
        if should_publish:
            self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = DiffusionInferenceNode()
    try:
        # 使用 MultiThreadedExecutor 可能更好，但简单的 Spin 也可以
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()