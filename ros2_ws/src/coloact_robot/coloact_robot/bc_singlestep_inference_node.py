#!/usr/bin/env python3
import sys
import os
import time
import threading
import math
import numpy as np
import cv2
import subprocess
CONDA_ENV_PATH = "/path/to/conda/env"
def main_with_conda():
    # 鎸囧悜 Conda 鐜閲岀殑 Python 瑙ｉ噴鍣?
    python_executable = f"{CONDA_ENV_PATH}/bin/python3.8"
    
    # 濡傛灉鎵句笉鍒拌繖涓?Python锛屽氨鏀惧純鍒囨崲锛岀洿鎺ュ穿鎺?
    if not os.path.exists(python_executable): 
        return

    # 璁剧疆鐜鍙橀噺锛岀‘淇濊兘鎵惧埌搴?
    env = os.environ.copy()
    env["PATH"] = f"{CONDA_ENV_PATH}/bin:{env['PATH']}"
    # 杩欎竴姝ュ緢鍏抽敭锛氭妸 Conda 鐨?site-packages 鍔犲埌 PYTHONPATH
    env["PYTHONPATH"] = f"{CONDA_ENV_PATH}/lib/python3.8/site-packages:{env.get('PYTHONPATH', '')}"
    
    # 鑾峰彇褰撳墠鑴氭湰鐨勭粷瀵硅矾寰?
    current_script = os.path.abspath(__file__)
    
    # 浣跨敤鏂扮殑 Python 閲嶆柊杩愯鑷繁
    subprocess.run([python_executable, current_script] + sys.argv[1:], env=env)

# 灏濊瘯瀵煎叆 torch锛屽鏋滃け璐ヨ鏄庢槸鍦ㄧ郴缁?Python 閲?
try:
    import torch
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool
    from coloact_robot.model_bc_singlestep import ColonBCSingleStep  # 瀵煎叆鍗曟妯″瀷
except ImportError:
    # 鎹曡幏鍒伴敊璇紝璇存槑鐜涓嶅锛岀珛鍒诲垏鎹紒
    main_with_conda()
    sys.exit(0)

class BCSingleStepInferenceNode(Node):
    def __init__(self):
        super().__init__('bc_singlestep_inference')
        self.get_logger().info("馃殌 Single-Step BC Pilot 鍚姩...")

        # === 鍙傛暟閰嶇疆 ===
        self.declare_parameter('input_mode', 'rgbd_elev') 
        self.declare_parameter('model_path', '/path/to/ros2_ws/src/coloact_robot/coloact_robot/Checkpoints/bc_singlestep_final.pth')
        self.declare_parameter('auto_start', False)

        self.input_mode = self.get_parameter('input_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.ai_enabled = self.get_parameter('auto_start').value

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.control_rate = 15.0 
        
        channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
        self.input_channels = channel_map.get(self.input_mode, 5)

        # === 鍔犺浇妯″瀷 ===
        if self.model_path == '':
            self.get_logger().error("鉂?璇峰湪 Launch 鏂囦欢涓寚瀹?model_path!")
            sys.exit(1)

        try:
            self.model = ColonBCSingleStep(input_channels=self.input_channels).to(self.device)
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.get_logger().info(f"鉁?妯″瀷鍔犺浇鎴愬姛: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"鉂?妯″瀷鍔犺浇澶辫触: {e}")
            self.model = None

        # === 鍙橀噺 ===
        self.current_cmd = [0.0, 0.0] # [v, w]
        self.last_rgb = None
        self.last_depth = None
        self.last_elev = None
        self.lock = threading.Lock()

        # === 閫氫俊 ===
        self.create_subscription(Bool, '/autonomous_enabled', self.enable_cb, 10)
        self.create_subscription(Image, '/camera/raw', self.rgb_cb, 10)
        self.create_subscription(Image, '/processed_depth', self.depth_cb, 10)
        self.create_subscription(Image, '/processed_elevation', self.elev_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / self.control_rate, self.loop)

    def enable_cb(self, msg): self.ai_enabled = msg.data
    def rgb_cb(self, msg): self.last_rgb = msg
    def depth_cb(self, msg): self.last_depth = msg
    def elev_cb(self, msg): self.last_elev = msg

    def decode_img(self, msg, encoding):
        if msg is None: return None
        dtype = np.uint16 if encoding == 'u16' else np.uint8
        shape = (msg.height, msg.width) if encoding != 'rgb' else (msg.height, msg.width, 3)
        return np.frombuffer(msg.data, dtype=dtype).reshape(shape)

    def loop(self):
        if self.last_rgb is None: return 
        
        # 瑙ｇ爜鍥惧儚
        img_rgb = self.decode_img(self.last_rgb, 'rgb')
        img_depth = self.decode_img(self.last_depth, 'u16')
        img_elev = self.decode_img(self.last_elev, 'u8')
        
        if img_rgb is None: return
        img_rgb = cv2.resize(img_rgb, (320, 240))
        if img_depth is not None: img_depth = cv2.resize(img_depth, (320, 240), interpolation=cv2.INTER_NEAREST)
        if img_elev is not None: img_elev = cv2.resize(img_elev, (320, 240), interpolation=cv2.INTER_NEAREST)

        # 鎺ㄧ悊
        self.inference(img_rgb, img_depth, img_elev)

        # 鍙鍖?& 鍙戝竷
        self.visualize(img_rgb, img_depth, img_elev)
        self.publish_cmd()

    def inference(self, rgb, depth, elev):
        # 褰掍竴鍖栦笌鍫嗗彔 (涓?Dataset 淇濇寔涓€鑷?
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        
        obs_list = [rgb_t]
        if 'd' in self.input_mode and depth is not None:
            depth_t = torch.from_numpy(depth).float() / 65535.0
            obs_list.append(depth_t.unsqueeze(0))
        if 'elev' in self.input_mode and elev is not None:
            elev_t = torch.from_numpy(elev).float() / 255.0
            obs_list.append(elev_t.unsqueeze(0))
            
        obs = torch.cat(obs_list, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 鍗曟妯″瀷鐩存帴杈撳嚭 [1, 2]
            action = self.model(obs) 
            self.current_cmd = action.cpu().numpy()[0] # [v, w]

    def visualize(self, img_rgb, img_depth, img_elev):
        # 绠€鍗曞彲瑙嗗寲锛岀粯鍒跺綋鍓嶇殑杞悜鎸囦护
        disp = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        v, w = self.current_cmd
        
        # 鍦ㄧ敾闈腑蹇冪敾涓€涓澶磋〃绀烘剰鍥?
        h, w_img = disp.shape[:2]
        cx, cy = w_img // 2, h - 20
        # 绠€鍗曠殑鏄犲皠锛歸鎺у埗宸﹀彸锛寁鎺у埗闀跨煭
        end_x = int(cx - w * 100)
        end_y = int(cy - v * 200)
        
        color = (0, 0, 255) # 绾㈣壊绠ご
        if self.ai_enabled: color = (0, 255, 0) # 缁胯壊绠ご
            
        cv2.arrowedLine(disp, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)
        
        cv2.putText(disp, "Single-Step BC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("BC Pilot HUD", disp)
        cv2.waitKey(1)

    def publish_cmd(self):
        cmd = Twist()
        if self.ai_enabled:
            # 閫熷害鏀惧ぇ (鏍规嵁浣犱箣鍓嶇殑鏁版嵁褰掍竴鍖栨儏鍐佃皟鏁?
            scale = 3.0
            cmd.linear.x = float(np.clip(self.current_cmd[0] * scale, -0.5, 0.5))
            cmd.angular.z = float(np.clip(self.current_cmd[1] * scale, -1.5, 1.5))
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = BCSingleStepInferenceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()


