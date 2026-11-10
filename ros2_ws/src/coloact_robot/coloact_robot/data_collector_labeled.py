#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
import message_filters
import h5py
import numpy as np
import os
import datetime
from threading import Lock
import cv2 

class DataCollectorLabeled(Node):
    def __init__(self):
        super().__init__('data_collector_labeled')
        
        # --- 1. 閰嶇疆瀛樺偍璺緞 ---
        self.save_dir = os.path.join(os.getcwd(), 'dataset_labeled')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # --- 2. 缂撳瓨鍖?(鏂板 rgb) ---
        self.buffer_size = 1000 
        self.data_buffer = {
            'rgb': [],      # [鏂板] 瀛樺偍 RGB 鍥惧儚
            'depth': [],
            'elevation': [],
            'action': [],
            'skill': []
        }
        self.lock = Lock()
        
        # --- 3. 鐘舵€佸彉閲?---
        self.current_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.current_skill = 0
        self.total_frames = 0
        self.is_recording = False
        
        # 瑙嗛褰曞埗鐩稿叧
        self.video_writer = None
        self.vlm_video_writer = None   # [鏂板] 鐢ㄤ簬褰曞埗楂樻竻 RGB (VLM鐢?


        self.temp_video_path = os.path.join(self.save_dir, ".temp_recording.avi")
        self.temp_vlm_video_path = os.path.join(self.save_dir, ".temp_vlm_raw.avi") # [鏂板]
        
        # 鍙鍖栫紦瀛?
        self.latest_raw = None          
        self.latest_colored_elev = None 
        
        # --- 4. 璁㈤槄璇濋 ---
        # 鎺у埗鎸囦护璁㈤槄
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(Int8, '/skill_id', self.skill_callback, 10)
        self.create_subscription(Joy, 'joy', self.joy_control_callback, 10)
        
        # 鍙鍖栦笓鐢?(浠呯敤浜庡疄鏃剁湅锛屼笉鐢ㄤ簬鍚屾褰曞埗)
        self.create_subscription(Image, '/colored_elevation', self.colored_elev_cb, 10)
        
        # --- [鍏抽敭淇敼] 澶氭ā鎬佸悓姝ヨ闃?(RGB + Depth + Elev) ---
        # 娉ㄦ剰锛氳繖閲屾垜浠妸 /camera/raw 涔熸斁鍏ュ悓姝ュ櫒
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/processed_depth')
        self.elev_sub = message_filters.Subscriber(self, Image, '/processed_elevation')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.elev_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info(f"澶氭ā鎬佹暟鎹噰闆嗚妭鐐瑰惎鍔?(RGB+Depth+Elev)銆備繚瀛樿矾寰? {self.save_dir}")

    def joy_control_callback(self, msg):
        """澶勭悊褰曞埗寮€鍏?(Start/Back 閿?"""
        START_BTN = 7 
        BACK_BTN = 6
        if len(msg.buttons) > START_BTN and msg.buttons[START_BTN] == 1:
            if not self.is_recording:
                self.start_recording()
        elif len(msg.buttons) > BACK_BTN and msg.buttons[BACK_BTN] == 1:
            if self.is_recording:
                self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.get_logger().info("馃敶 褰曞埗寮€濮?")
        # 640x480 鏄彲瑙嗗寲鎷煎浘鍚庣殑鎬诲ぇ灏?
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 1. 鍥涘鏍艰棰?(640x480) - 淇濇寔涓嶅彉
        self.video_writer = cv2.VideoWriter(self.temp_video_path, fourcc, 15.0, (640, 480))
        # 2. [鏂板] VLM 涓撶敤楂樻竻瑙嗛
        # 娉ㄦ剰锛氳繖閲岀殑鍒嗚鲸鐜?(640, 480) 蹇呴』涓庝綘鐩告満瀹為檯鍙戝竷鐨?/camera/raw 灏哄涓€鑷达紒
        # 濡傛灉浣犲湪娣卞害鑺傜偣璁句簡 1280x720锛岃繖閲屽氨瑕佹敼鎴?(1280, 720)
        self.vlm_video_writer = cv2.VideoWriter(self.temp_vlm_video_path, fourcc, 15.0, (640, 480))

    def stop_recording(self):
        self.is_recording = False
        self.get_logger().info("猬?褰曞埗鍋滄锛屾鍦ㄤ繚瀛?..")

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.vlm_video_writer: # [鏂板]
            self.vlm_video_writer.release()
            self.vlm_video_writer = None

        self.flush_buffer()

    def colored_elev_cb(self, msg):
        try:
            if msg.encoding == 'bgr8':
                self.latest_colored_elev = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except: pass

    def cmd_callback(self, msg):
        self.current_vel = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    def skill_callback(self, msg):
        self.current_skill = msg.data

    def sync_callback(self, rgb_msg, depth_msg, elev_msg):
        """鍚屾鍥炶皟锛氬悓鏃舵帴鏀?RGB銆佹繁搴﹀拰楂樼▼"""
        try:
            # 1. 瑙ｆ瀽 RGB 骞?Resize 鍒?320x240
            rgb_raw_hd = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape((rgb_msg.height, rgb_msg.width, 3))
            # 寮哄埗 Resize 浠ュ尮閰嶆繁搴﹀浘 (娣卞害鍥炬槸 320x240)
            rgb_resized = cv2.resize(rgb_raw_hd, (320, 240))
            # 褰掍竴鍖?RGB [0, 1]
            rgb_f32 = rgb_resized.astype(np.float32) / 255.0
            
            # 鏇存柊鐢ㄤ簬鍙鍖栫殑 latest_raw (淇濇寔鍘熷灏哄鎴朢esize鍧囧彲锛岃繖閲岀敤resized鏂逛究鎷煎浘)
            self.latest_raw = rgb_resized

            # 2. 瑙ｆ瀽 Depth (16bit -> float)
            depth_raw = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((depth_msg.height, depth_msg.width))
            depth_f32 = depth_raw.astype(np.float32) / 65535.0
            
            # 3. 瑙ｆ瀽 Elevation (8bit)
            elev_raw = np.frombuffer(elev_msg.data, dtype=np.uint8).reshape((elev_msg.height, elev_msg.width))

            # 4. 瀛樻暟鎹?
            if self.is_recording:
                # 澧炲姞缁村害鍙樻垚 (H, W, 1) 鏂逛究鍚庣画澶勭悊
                depth_save = depth_f32[..., np.newaxis]
                elev_save = elev_raw[..., np.newaxis]
                # RGB 宸茬粡鏄?(H, W, 3)

                with self.lock:
                    self.data_buffer['rgb'].append(rgb_f32) # 瀛樺叆褰掍竴鍖栫殑 RGB
                    self.data_buffer['depth'].append(depth_save)
                    self.data_buffer['elevation'].append(elev_save)
                    self.data_buffer['action'].append(self.current_vel.copy())
                    self.data_buffer['skill'].append(self.current_skill)

                # 2. 銆愬氨鍦ㄨ繖閲屾彃鍏ヤ唬鐮侊紒銆?
                # 鏀惧湪 H5 鏁版嵁瀛樺畬涔嬪悗锛屾垨鑰?if self.is_recording: 鐨勭涓€琛岄兘鍙互
                # 纭繚鍙湪褰曞埗鏃跺啓鍏ワ紝鍚﹀垯浼氱敓鎴愬法澶х殑鏃犳晥瑙嗛
                if self.vlm_video_writer is not None:
                    self.vlm_video_writer.write(rgb_raw_hd)
                    
                self.total_frames += 1
                if self.total_frames % 30 == 0:
                    print(f"\r姝ｅ湪褰曞埗... 甯ф暟: {len(self.data_buffer['depth'])}", end="")

            # 5. 鍙鍖?
            self.visualize(rgb_resized, depth_f32, elev_raw)

        except Exception as e:
            self.get_logger().error(f"澶勭悊鏁版嵁澶辫触: {e}")

    def visualize(self, img_raw, depth_f32, elev_raw):
        W, H = 320, 240
        
        # --- 1. 鍒涘缓涓€涓嫭绔嬬殑鈥滅姸鎬佺獥鍙ｂ€濈敤浜庢樉绀烘枃瀛?---
        # 鍒涘缓涓€涓珮100锛屽400鐨勯粦鑹茶儗鏅浘
        status_img = np.zeros((100, 400, 3), dtype=np.uint8)
        
        # 鍑嗗鏂囧瓧鍐呭
        if self.is_recording:
            status_text = f"REC [ON] | Frames: {self.total_frames}"
            color = (0, 0, 255) # 绾㈣壊
            # 鍦ㄧ姸鎬佺獥鍙ｇ敾绾㈢偣
            cv2.circle(status_img, (20, 50), 10, color, -1)
        else:
            status_text = "REC [OFF] | PAUSED"
            color = (200, 200, 200) # 鐏拌壊
            
        skill_text = f"Current Skill: {self.current_skill}"
        
        # 灏嗘枃瀛楀啓鍦?status_img 涓婏紝鑰屼笉鏄?img_raw 涓?
        cv2.putText(status_img, status_text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(status_img, skill_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 鏄剧ず鐘舵€佺獥鍙?
        cv2.imshow("Status Monitor (Text Only)", status_img)

        # --- 2. 澶勭悊涓荤敾闈?(淇濇寔绾噣) ---
        # 娣卞害浼僵鑹?
        depth_u8 = (depth_f32 * 255).astype(np.uint8)
        img_depth = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
        
        # 楂樼▼鍥?(鐏板害杞珺GR)
        img_elev_gray = cv2.cvtColor(elev_raw, cv2.COLOR_GRAY2BGR)
        
        # 褰╄壊楂樼▼
        if self.latest_colored_elev is not None:
            img_elev_color = cv2.resize(self.latest_colored_elev, (W, H))
        else:
            img_elev_color = np.zeros((H, W, 3), dtype=np.uint8)

        # 鎷兼帴 2x2 缃戞牸 (姝ゆ椂 img_raw 涓婃病鏈変换浣曟枃瀛?
        top_row = np.hstack((img_raw, img_depth))
        bottom_row = np.hstack((img_elev_gray, img_elev_color))
        grid = np.vstack((top_row, bottom_row))

        # --- 3. 瑙嗛褰曞埗閫昏緫 ---
        # 濡傛灉寮€鍚簡鍐呴儴褰曞埗锛岃繖閲屽綍杩涘幓鐨勭敾闈篃鏄病鏂囧瓧鐨?(绗﹀悎浣犵殑瑕佹眰)
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(grid)

        # 鏄剧ず涓荤獥鍙?
        cv2.imshow("Multi-Modal Data Collector (Clean)", grid)
        cv2.waitKey(1)

    def flush_buffer(self):
        """淇濆瓨鏁版嵁鍒?H5 鍜?涓や釜瑙嗛鏂囦欢"""
        # 濡傛灉缂撳瓨涓虹┖锛岀洿鎺ヨ繑鍥?
        if len(self.data_buffer['depth']) == 0:
            return

        # 1. 鐢熸垚鏂囦欢鍚嶅熀纭€閮ㄥ垎: skill{ID}_{鏃堕棿鎴硙
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 闃叉绌哄垪琛ㄦ姤閿欙紝铏界劧鍓嶉潰妫€鏌ヤ簡闀垮害
        if len(self.data_buffer['skill']) > 0:
            skill_label = self.data_buffer['skill'][-1]
        else:
            skill_label = 0
            
        base_name = f"skill{skill_label}_{timestamp}"
        
        # 瀹氫箟瀹屾暣鐨勬枃浠惰矾寰?
        h5_filename = os.path.join(self.save_dir, f"{base_name}.h5")
        
        # 瀹氫箟涓や釜瑙嗛鐨勬渶缁堟枃浠跺悕
        # (1) 鍥涘鏍艰皟璇曡棰?(Grid)
        grid_avi_filename = os.path.join(self.save_dir, f"{base_name}_grid.avi")      
        # (2) VLM涓撶敤楂樻竻鍘熷浘瑙嗛 (Raw)
        vlm_avi_filename = os.path.join(self.save_dir, f"{base_name}_vlm_raw.avi")   
        
        self.get_logger().info(f"姝ｅ湪淇濆瓨 H5 鏂囦欢: {h5_filename} ...")
        
        try:
            with self.lock:
                # 杞崲涓?numpy 鏁扮粍
                rgbs = np.array(self.data_buffer['rgb'], dtype=np.float32) 
                depths = np.array(self.data_buffer['depth'], dtype=np.float32)
                elevs = np.array(self.data_buffer['elevation'], dtype=np.uint8)
                actions = np.array(self.data_buffer['action'], dtype=np.float32)
                skills = np.array(self.data_buffer['skill'], dtype=np.int8)
                
                # 娓呯┖缂撳瓨 & 閲嶇疆璁℃暟
                self.data_buffer = {'rgb': [], 'depth': [], 'elevation': [], 'action': [], 'skill': []}
                self.total_frames = 0
            
            # --- 2. 淇濆瓨 H5 鏂囦欢 ---
            with h5py.File(h5_filename, 'w') as f:
                f.create_dataset('rgb', data=rgbs, compression="gzip")
                f.create_dataset('depth', data=depths, compression="gzip")
                f.create_dataset('elevation', data=elevs, compression="gzip")
                f.create_dataset('action', data=actions)
                f.create_dataset('skill', data=skills)
            
            # --- 3. 澶勭悊瑙嗛鏂囦欢 ---
            
            # (A) 淇濆瓨鍥涘鏍艰棰?
            if os.path.exists(self.temp_video_path):
                os.rename(self.temp_video_path, grid_avi_filename)
                self.get_logger().info(f"鍥涘鏍艰棰戝凡淇濆瓨: {grid_avi_filename}")
            
            # (B) [鏂板] 淇濆瓨 VLM 楂樻竻鍘熷浘瑙嗛
            if os.path.exists(self.temp_vlm_video_path):
                os.rename(self.temp_vlm_video_path, vlm_avi_filename)
                self.get_logger().info(f"鉁?VLM涓撶敤楂樻竻瑙嗛宸蹭繚瀛? {vlm_avi_filename}")
                
            self.get_logger().info(">>> 鍏ㄩ儴鏁版嵁淇濆瓨鎴愬姛锛?<<<")
            
        except Exception as e:
            self.get_logger().error(f"淇濆瓨澶辫触: {e}")

    def destroy_node(self):
        if self.is_recording:
            self.stop_recording()
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = DataCollectorLabeled()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


