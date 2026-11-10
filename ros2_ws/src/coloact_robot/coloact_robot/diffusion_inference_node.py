#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import threading
from collections import deque
import math
import copy # 寮曞叆 copy 搴?

# =========================================================
# 馃敟 鑷姩鐜鍒囨崲 (淇濇寔涓嶅彉)
# =========================================================
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
    import numpy as np
    import cv2
    from coloact_robot.model_diffusion import ColonDiffusionPolicy
except ImportError:
    main_with_conda()
    sys.exit(0)

class DiffusionInferenceNode(Node):
    def __init__(self):
        super().__init__('diffusion_inference_node')
        self.get_logger().info("馃殌 Diffusion Pilot (Asynchronous + RHC) 鍚姩...")

        # === 1. 鍙傛暟閰嶇疆 ===
        self.declare_parameter('input_mode', 'rgbd_elev') 
        # 浣犵殑榛樿妯″瀷璺緞
        default_path = '/path/to/ros2_ws/src/coloact_robot/coloact_robot/Checkpoints/checkpoints_rgbd_elev/diffusion_final.pth'
        self.declare_parameter('model_path', default_path)     
        self.declare_parameter('auto_start', False)       

        self.input_mode = self.get_parameter('input_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.auto_start = self.get_parameter('auto_start').value

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 绛栫暐鍙傛暟
        self.pred_horizon = 16
        self.action_horizon = 8 # 鎵ц 8 姝ュ悗閲嶆柊鎺ㄧ悊 (Receding Horizon)
        self.control_rate = 15.0 
        
        channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
        self.input_channels = channel_map.get(self.input_mode, 5)

        # === 2. 鍔犺浇妯″瀷 ===
        try:
            self.model = ColonDiffusionPolicy(
                pred_horizon=self.pred_horizon, 
                input_channels=self.input_channels
            ).to(self.device)
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.get_logger().info(f"鉁?妯″瀷鍔犺浇鎴愬姛: {os.path.basename(self.model_path)}")
        except Exception as e:
            self.get_logger().error(f"鉂?妯″瀷鍔犺浇澶辫触: {e}")
            self.model = None

        # === 3. 鍙橀噺鍒濆鍖?===
        self.action_queue = deque(maxlen=self.pred_horizon)
        self.current_trajectory = [] # 鐢ㄤ簬鍙鍖?
        
        # 馃У 绾跨▼閿佷笌鏍囧織浣?(鍏抽敭淇敼)
        self.lock = threading.Lock()
        self.is_inferencing = False 
        
        self.ai_enabled = self.auto_start
        self.last_rgb = None
        self.last_depth = None
        self.last_elev = None
        
        # 鎬ц兘缁熻
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.latency = 0.0

        # === 4. 閫氫俊鎺ュ彛 ===
        self.create_subscription(Bool, '/autonomous_enabled', self.enable_cb, 10)
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/camera/raw', self.rgb_cb, qos)
        self.create_subscription(Image, '/processed_depth', self.depth_cb, qos)
        self.create_subscription(Image, '/processed_elevation', self.elev_cb, qos)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / self.control_rate, self.loop)
        self.output_ema_alpha = 0.2 
        self.current_v = 0.0
        self.current_w = 0.0

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
        # --- FPS 缁熻 ---
        self.frame_count += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (time.time() - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = time.time()

        if self.last_rgb is None: return 
        
        # --- 瑙ｇ爜鍥惧儚 ---
        img_rgb = self.decode_img(self.last_rgb, 'rgb')
        img_depth = self.decode_img(self.last_depth, 'u16') 
        img_elev = self.decode_img(self.last_elev, 'u8')

        if img_rgb is None: return
        img_rgb = cv2.resize(img_rgb, (320, 240))
        if img_depth is not None: img_depth = cv2.resize(img_depth, (320, 240), interpolation=cv2.INTER_NEAREST)
        if img_elev is not None: img_elev = cv2.resize(img_elev, (320, 240), interpolation=cv2.INTER_NEAREST)

        # =========================================================
        # 馃敟 淇敼鐐癸細寮傛闈為樆濉炴帹鐞?
        # =========================================================
        # 绛栫暐锛氬綋闃熷垪閲岀殑鍔ㄤ綔蹇笉澶熶簡 (<= 4)锛屼笖鍚庡彴娌℃湁鍦ㄧ畻锛屽氨鍚姩鏂扮嚎绋?
        with self.lock:
            queue_len = len(self.action_queue)
        
        if queue_len <= 4 and not self.is_inferencing:
            # 蹇呴』娣辨嫹璐濆浘鍍忥紝闃叉绾跨▼绔炰簤瀵艰嚧鏁版嵁鎹熷潖
            t_rgb = img_rgb.copy()
            t_depth = img_depth.copy() if img_depth is not None else None
            t_elev = img_elev.copy() if img_elev is not None else None
            
            # 鍚姩鍚庡彴绾跨▼
            thread = threading.Thread(target=self.run_inference_thread, args=(t_rgb, t_depth, t_elev))
            thread.start()

        # --- 鍙鍖?& 鍙戝竷鎸囦护 (杩欎竴姝ユ案杩滀笉鍗￠】) ---
        self.visualize(img_rgb, img_depth, img_elev)
        self.publish_cmd()

    def run_inference_thread(self, rgb, depth, elev):
        """鍚庡彴绾跨▼鍏ュ彛"""
        self.is_inferencing = True
        try:
            self.inference(rgb, depth, elev)
        except Exception as e:
            self.get_logger().error(f"鍚庡彴鎺ㄧ悊寮傚父: {e}")
        finally:
            self.is_inferencing = False

    def inference(self, rgb, depth, elev):
        t0 = time.time()
        
        # --- 1. 鏁版嵁棰勫鐞?---
        # 褰掍竴鍖?RGB [0, 255] -> [0, 1]
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0 
        
        depth_t = None
        if depth is not None:
            # 褰掍竴鍖?Depth [0, 65535] -> [0, 1]
            depth_t = torch.from_numpy(depth).float() / 65535.0
            depth_t = depth_t.unsqueeze(0) # [1, H, W]

        elev_t = None
        if elev is not None:
            # 褰掍竴鍖?Elev [0, 255] -> [0, 1]
            elev_t = torch.from_numpy(elev).float() / 255.0
            elev_t = elev_t.unsqueeze(0) # [1, H, W]

        # 鍫嗗彔寮犻噺
        obs = None
        if self.input_mode == 'rgb': obs = rgb_t
        elif self.input_mode == 'rgbd': obs = torch.cat([rgb_t, depth_t], dim=0)
        elif self.input_mode == 'rgbd_elev': obs = torch.cat([rgb_t, depth_t, elev_t], dim=0)
        
        if obs is None: return
        obs = obs.unsqueeze(0).to(self.device)

        # --- 2. 妯″瀷鎺ㄧ悊 (GPU 鑰楁椂鎿嶄綔) ---
        with torch.no_grad():
            action_traj = self.model.predict_action(obs)
            actions = action_traj.cpu().numpy()[0] # [16, 2]

        # --- 3. 鏇存柊鍔ㄤ綔闃熷垪 (鍔犻攣) ---
        with self.lock:
            self.current_trajectory = actions.copy() # 鏇存柊鍙鍖栨暟鎹?
            
            # Receding Horizon Control:
            # 鎴戜滑鐩存帴娓呯┖鏃х殑娈嬩綑鍔ㄤ綔锛屽～鍏ユ渶鏂扮殑楂樿川閲忛娴?
            # 鍥犱负鎺ㄧ悊鏄紓姝ョ殑锛屼富绾跨▼鍙兘鍦ㄦ帹鐞嗘湡闂村凡缁忔秷鑰椾簡涓€浜涘姩浣?
            self.action_queue.clear()
            
            # 鍙～鍏ュ墠 action_horizon 姝?(渚嬪 8 姝?
            # 杩欐牱淇濊瘉浜嗘垜浠€绘槸浣跨敤姣忔潯杞ㄨ抗鏈€鍑嗙‘鐨勨€滃ご閮ㄢ€?
            for a in actions[:self.action_horizon]:
                self.action_queue.append(a)
        
        self.latency = (time.time() - t0) * 1000

    # =========================================================
    # 馃帹 鍙鍖栦笌杞ㄨ抗缁樺埗
    # =========================================================
    def draw_trajectory(self, img, actions):
        """鍦ㄥ浘鍍忎笂缁樺埗棰勬祴鐨勮建杩圭嚎"""
        h, w_img = img.shape[:2]
        cx, cy = w_img // 2, h 
        
        # 杩愬姩瀛︾Н鍒嗗垵濮嬪寲
        theta = 0.0
        dt = 0.1 # 鏃堕棿姝ラ暱
        viz_scale = 5.0 # 瑙嗚鏀惧ぇ鍊嶆暟
        
        points_2d = []
        
        # 璁＄畻杞ㄨ抗鐐瑰儚绱犲潗鏍?
        for v, omega in actions:
            v_scaled = v * viz_scale
            # 绠€鍗曠殑鎶曞奖妯″瀷妯℃嫙
            step_y = v_scaled * 20.0  # 鍓嶈繘鍒嗛噺
            step_x = -omega * 20.0    # 杞悜鍒嗛噺
            
            cy -= step_y * np.cos(theta)
            cx += step_x + step_y * np.sin(theta)
            theta += omega * dt
            
            points_2d.append((int(cx), int(cy)))

        # 缁樺埗杩炵嚎
        if len(points_2d) > 1:
            for i in range(len(points_2d) - 1):
                pt1 = points_2d[i]
                pt2 = points_2d[i+1]
                # 绾挎潯绮楃粏娓愬彉 (杩戝绮楋紝杩滃缁?
                thickness = max(1, int(4 - i * 0.2))
                # 缁胯壊杞ㄨ抗
                cv2.line(img, pt1, pt2, (0, 255, 0), thickness)
                if i % 4 == 0: # 姣忛殧鍑犱釜鐐圭敾涓渾鍦?
                    cv2.circle(img, pt1, thickness+1, (0, 255, 0), -1)

    def visualize(self, img_rgb, img_depth, img_elev):
        # 1. 鍑嗗 RGB 搴曞浘
        disp_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 2. 鍑嗗娣卞害搴曞浘 (褰掍竴鍖栨樉绀?
        disp_depth = np.zeros_like(disp_rgb)
        if img_depth is not None:
            # 0-65535 -> 0-255
            d_u8 = (img_depth / 256.0).astype(np.uint8)
            disp_depth = cv2.applyColorMap(d_u8, cv2.COLORMAP_MAGMA)

        # 3. 鍑嗗楂樼▼搴曞浘 (榛戠櫧鏄剧ず鍑犱綍缁撴瀯鏇存竻鏅?
        disp_elev = np.zeros_like(disp_rgb)
        if img_elev is not None:
            # 0-255 -> 0-255
            disp_elev = cv2.cvtColor(img_elev, cv2.COLOR_GRAY2BGR)

        # 4. 缁樺埗杞ㄨ抗 (绾跨▼瀹夊叏璇诲彇)
        traj_to_draw = []
        with self.lock:
            if len(self.current_trajectory) > 0:
                traj_to_draw = self.current_trajectory.copy()
        
        if len(traj_to_draw) > 0:
            self.draw_trajectory(disp_rgb, traj_to_draw)
            self.draw_trajectory(disp_depth, traj_to_draw)
            self.draw_trajectory(disp_elev, traj_to_draw)

        # 5. 鎷兼帴 HUD 淇℃伅闈㈡澘
        info_panel = np.zeros_like(disp_rgb)
        
        # 鐘舵€佹枃瀛?
        txt_state = "AUTO PILOT" if self.ai_enabled else "STANDBY (Hold L1)"
        color_state = (0, 255, 0) if self.ai_enabled else (0, 0, 255)
        
        cv2.putText(info_panel, txt_state, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_state, 2)
        cv2.putText(info_panel, f"Queue: {len(self.action_queue)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(info_panel, f"FPS: {self.fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Infer: {self.latency:.0f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # 2x2 缃戞牸甯冨眬
        row1 = np.hstack((disp_rgb, disp_depth))
        row2 = np.hstack((disp_elev, info_panel))
        grid = np.vstack((row1, row2))
        
        cv2.imshow("Diffusion Pilot HUD (Async)", grid)
        cv2.waitKey(1)

    def publish_cmd(self):
        cmd = Twist()
        should_move = False

        if self.ai_enabled:
            # 绾跨▼瀹夊叏鍦板彇鍔ㄤ綔
            with self.lock:
                if len(self.action_queue) > 0:
                    action = self.action_queue.popleft()

                    # 1. 鑾峰彇鍘熷妯″瀷杈撳嚭
                    target_v = float(action[0]) * 3.0 # 鍋囪浣犵殑 speed_scale 鏄?3.0
                    target_w = float(action[1]) * 3.0

                    # 2. 馃敟 鏍稿績淇敼锛氳緭鍑虹 EMA 婊ゆ尝 馃敟
                    # 鍏紡: current = alpha * target + (1-alpha) * current
                    self.current_v = self.output_ema_alpha * target_v + (1 - self.output_ema_alpha) * self.current_v
                    self.current_w = self.output_ema_alpha * target_w + (1 - self.output_ema_alpha) * self.current_w

                    # 3. 璧嬪€?(骞跺仛闄愬箙)
                    cmd.linear.x = float(np.clip(self.current_v, -0.5, 0.5))
                    cmd.angular.z = float(np.clip(self.current_w, -1.5, 1.5))
                    should_move = True

        if not should_move:
            # 濡傛灉鍋滄锛屼篃瑕佹參鎱㈠仠涓嬫潵锛屼笉瑕佺灛闂村綊闆?(Soft Stop)
            self.current_v = self.current_v * 0.5
            self.current_w = self.current_w * 0.5
            cmd.linear.x = self.current_v
            cmd.angular.z = self.current_w
            if abs(self.current_v) < 0.01: self.current_v = 0.0
            if abs(self.current_w) < 0.01: self.current_w = 0.0

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = DiffusionInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


