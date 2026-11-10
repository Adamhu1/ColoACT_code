#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import threading
from collections import deque
import numpy as np
import copy

# 鐜鍒囨崲 (澶嶇敤涔嬪墠鐨?
CONDA_ENV_PATH = "/path/to/conda/env"
def main_with_conda():
    python_executable = f"{CONDA_ENV_PATH}/bin/python3.8"
    if not os.path.exists(python_executable): return
    env = os.environ.copy()
    env["PATH"] = f"{CONDA_ENV_PATH}/bin:{env['PATH']}"
    env["PYTHONPATH"] = f"{CONDA_ENV_PATH}/lib/python3.8/site-packages:{env.get('PYTHONPATH', '')}"
    current_script = os.path.abspath(__file__)
    subprocess.run([python_executable, current_script] + sys.argv[1:], env=env)

try:
    import torch
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool
    import cv2
    from coloact_robot.model_act import ACTPolicy # 瀵煎叆 ACT 妯″瀷
except ImportError:
    main_with_conda()
    sys.exit(0)

class ACTInferenceNode(Node):
    def __init__(self):
        super().__init__('act_inference_node')
        self.get_logger().info("馃殌 ACT Pilot (Temporal Ensembling) 鍚姩...")

        # 鍙傛暟
        self.declare_parameter('input_mode', 'rgbd_elev')
        self.declare_parameter('model_path', '/path/to/ros2_ws/src/coloact_robot/coloact_robot/Checkpoints_ACT/act_final.pth')
        self.declare_parameter('auto_start', False)

        self.input_mode = self.get_parameter('input_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.ai_enabled = self.get_parameter('auto_start').value
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_horizon = 16
        self.control_rate = 15.0

        channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
        
        # 鍔犺浇妯″瀷
        try:
            self.model = ACTPolicy(
                pred_horizon=self.pred_horizon,
                input_channels=channel_map[self.input_mode]
            ).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f"鉁?ACT 妯″瀷鍔犺浇鎴愬姛")
        except Exception as e:
            self.get_logger().error(f"鉂?妯″瀷閿欒: {e}")
            sys.exit(1)

        # === 鏍稿績锛氭椂搴忛泦鎴愮紦鍐插尯 ===
        # 杩欎釜 Buffer 鐢ㄤ簬瀛樺偍杩囧幓鍑犳鎺ㄧ悊瀵规湭鏉ュ姩浣滅殑鍙犲姞
        # 鍋囪 max_timesteps 瓒冲澶э紝瀹為檯涓婃垜浠彧闇€瑕佺淮鎶ゆ粴鍔ㄧ獥鍙?
        self.action_buffer_size = 100 
        self.all_time_actions = np.zeros((self.action_buffer_size, self.pred_horizon, 2))
        self.all_time_weights = np.zeros((self.action_buffer_size, self.pred_horizon))
        self.time_idx = 0 # 褰撳墠鏃堕棿姝ョ储寮?(寰幆浣跨敤)

        self.lock = threading.Lock()
        self.is_inferencing = False
        
        # 鍥惧儚缂撳瓨
        self.last_rgb = None
        self.last_depth = None
        self.last_elev = None

        # 閫氫俊
        self.create_subscription(Bool, '/autonomous_enabled', self.enable_cb, 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/camera/raw', self.rgb_cb, qos)
        self.create_subscription(Image, '/processed_depth', self.depth_cb, qos)
        self.create_subscription(Image, '/processed_elevation', self.elev_cb, qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0/self.control_rate, self.loop)

    def enable_cb(self, msg): self.ai_enabled = msg.data
    def rgb_cb(self, msg): self.last_rgb = msg
    def depth_cb(self, msg): self.last_depth = msg
    def elev_cb(self, msg): self.last_elev = msg

    def decode_img(self, msg, encoding):
        if msg is None: return None
        try:
            if encoding == 'rgb': return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            elif encoding == 'u16': return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            elif encoding == 'u8': return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        except: return None
        return None

    def loop(self):
        if self.last_rgb is None: return
        
        # 1. 鍑嗗鍥惧儚
        img_rgb = self.decode_img(self.last_rgb, 'rgb')
        img_depth = self.decode_img(self.last_depth, 'u16')
        img_elev = self.decode_img(self.last_elev, 'u8')
        if img_rgb is None: return
        
        img_rgb = cv2.resize(img_rgb, (320, 240))
        if img_depth is not None: img_depth = cv2.resize(img_depth, (320, 240), interpolation=cv2.INTER_NEAREST)
        if img_elev is not None: img_elev = cv2.resize(img_elev, (320, 240), interpolation=cv2.INTER_NEAREST)

        # 2. 鍚姩寮傛鎺ㄧ悊
        # ACT 姣忎竴甯ч兘瑕佹帹鐞嗭紝浠ヤ究杩涜闆嗘垚
        if not self.is_inferencing:
            t_rgb = img_rgb.copy()
            t_depth = img_depth.copy() if img_depth is not None else None
            t_elev = img_elev.copy() if img_elev is not None else None
            threading.Thread(target=self.run_inference, args=(t_rgb, t_depth, t_elev)).start()

        # 3. 鍙戝竷鍔ㄤ綔
        self.publish_act_cmd()
        
        # 绠€鍗曞彲瑙嗗寲
        cv2.imshow("ACT Pilot HUD", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def run_inference(self, rgb, depth, elev):
        self.is_inferencing = True
        try:
            # 褰掍竴鍖?
            rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
            depth_t = None
            if depth is not None: depth_t = torch.from_numpy(depth).float().unsqueeze(0) / 65535.0
            elev_t = None
            if elev is not None: elev_t = torch.from_numpy(elev).float().unsqueeze(0) / 255.0

            obs_list = [rgb_t]
            if depth_t is not None: obs_list.append(depth_t)
            if elev_t is not None: obs_list.append(elev_t)
            obs = torch.cat(obs_list, dim=0).unsqueeze(0).to(self.device)

            # ACT 鎺ㄧ悊: 浼犲叆 obs锛屼笉浼?action锛屾ā鍨嬩細鑷姩鐢?z=0
            with torch.no_grad():
                pred_actions, _, _ = self.model(obs, actions=None)
                actions = pred_actions.cpu().numpy()[0] # [16, 2]

            # === 鏃跺簭闆嗘垚鏇存柊 ===
            with self.lock:
                # 鍋囪褰撳墠缁濆鏃堕棿姝ユ槸 T
                # 鎴戜滑鎶婅繖涓娴嬫斁鍦?Buffer 鐨?T 浣嶇疆
                # buffer 绱㈠紩寰幆浣跨敤
                curr_idx = self.time_idx % self.action_buffer_size
                
                self.all_time_actions[curr_idx] = actions
                # 鏉冮噸绛栫暐锛氭寚鏁拌“鍑忔潈閲?exp(-k*m)锛岃繖閲岀畝鍖栦负鍧囧寑鏉冮噸 1.0
                # ACT 鍘熸枃浣跨敤鎸囨暟鏉冮噸锛屼负浜嗙畝鍗曡捣瑙侊紝鎴戜滑杩欓噷璁句负 1.0锛岃〃绀烘墍鏈夐噸鍙犻娴嬪悓绛夐噸瑕?
                self.all_time_weights[curr_idx] = 1.0 
                
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        finally:
            self.is_inferencing = False

    def publish_act_cmd(self):
        cmd = Twist()
        
        if self.ai_enabled:
            with self.lock:
                # === ACT 鏍稿績锛氳绠楀姞鏉冨钩鍧?===
                # 褰撳墠鏃堕棿姝?T 瀵瑰簲鐨勫姩浣滐紝鏄敱锛?
                # T 鏃跺埢棰勬祴鐨勭 0 姝?
                # T-1 鏃跺埢棰勬祴鐨勭 1 姝?
                # T-2 鏃跺埢棰勬祴鐨勭 2 姝?
                # ...
                # T-15 鏃跺埢棰勬祴鐨勭 15 姝?
                # 缁勬垚鐨勩€傛垜浠渶瑕佹妸杩欎簺鍏ㄩ儴鍔犳潈骞冲潎銆?
                
                raw_action = np.zeros(2)
                total_weight = 0.0
                
                # 閬嶅巻杩囧幓 pred_horizon 甯х殑棰勬祴
                for t_offset in range(self.pred_horizon):
                    # 閭ｄ釜鏃跺埢鍦?buffer 涓殑绱㈠紩
                    past_t = self.time_idx - t_offset
                    if past_t < 0: continue # 鍒氬惎鍔ㄦ椂鍘嗗彶涓嶈冻
                    
                    buff_idx = past_t % self.action_buffer_size
                    
                    # 鍙栧嚭閭ｄ釜鏃跺埢棰勬祴鐨勭 t_offset 姝?
                    act = self.all_time_actions[buff_idx, t_offset]
                    w = self.all_time_weights[buff_idx, t_offset]
                    
                    raw_action += act * w
                    total_weight += w
                
                if total_weight > 0:
                    raw_action /= total_weight
                    
                    # 閫熷害缂╂斁
                    scale = 3.0
                    cmd.linear.x = float(np.clip(raw_action[0] * scale, -0.5, 0.5))
                    cmd.angular.z = float(np.clip(raw_action[1] * scale, -1.5, 1.5))
                
                # 鏃堕棿姝ユ帹杩?
                self.time_idx += 1
                
                # 娓呯悊鏃ф潈閲?(Ring Buffer 瑕嗙洊鍓嶈娓呴浂)
                next_idx = (self.time_idx + 1) % self.action_buffer_size
                self.all_time_weights[next_idx] = 0.0

        else:
            # 涓嶅惎鐢ㄦ椂涔熻鎺ㄨ繘鏃堕棿绱㈠紩锛屾垨鑰呴噸缃?
            pass

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = ACTInferenceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()


