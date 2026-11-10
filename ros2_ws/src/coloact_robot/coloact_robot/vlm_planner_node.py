#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Int8
import numpy as np
import cv2
import requests
import base64
import json
import re

class VLMPlannerNode(Node):
    def __init__(self):
        super().__init__('vlm_planner_node')

        # --- 閰嶇疆鍖哄煙 ---
        # 1. VLM Server 鍦板潃 (鎸囧悜浣犵殑 Docker 瀹瑰櫒)
        # 濡傛灉 Docker 鍦ㄦ湰鏈猴紝閫氬父鏄?172.17.0.1 鎴?localhost
        self.server_url = "http://172.17.0.1:8000/plan"
        
        # 2. 瑙勫垝棰戠巼 (Hz)
        self.planning_rate = 1.0 
        
        # 3. 鏍稿績 Prompt (涓庝綘鐨?Skill ID 瀵瑰簲)
        self.system_prompt = """
        You are a colonoscopy robot navigation assistant operating in a pig colon. 
        Analyze the image and output the correct Skill ID (0, 1, 2, or 3).
        
        Decision Logic Chain (Step-by-Step):
        
        1. [HIGHEST PRIORITY] TARGET DETECTION:
           - Is there a RED MARKER or RED POLYP visible?
           -> YES: Output 1 (Inspect/Hover).
           
        2. [HIGH PRIORITY] EMERGENCY CHECK:
           - Is the view completely blocked by a pink wall (blur)?
           - Or is the camera clearly stuck/too close to tissue?
           -> YES: Output 3 (Recover/Back).
           
        3. [NORMAL PRIORITY] NAVIGATION GEOMETRY:
           - Is the lumen (dark hole) hidden by folds, or is there a sharp S-turn? 
           -> YES: Output 2 (Turn).
           
           - Is the lumen visible and centered, path looks relatively clear?
           -> YES: Output 0 (Straight).

        Constraint: Reply ONLY with the single digit (e.g., 0). Do not explain.
        """

        # --- 閫氫俊鎺ュ彛 ---
        self.create_subscription(RosImage, '/camera/raw', self.image_callback, 10)
        self.skill_pub = self.create_publisher(Int8, '/skill_id', 10)

        # --- 鐘舵€佸彉閲?---
        self.latest_image = None
        self.is_requesting = False # 浜掓枼閿侊紝闃叉璇锋眰鍫嗙Н
        
        # --- 鍚姩瀹氭椂鍣?---
        self.create_timer(1.0 / self.planning_rate, self.planning_loop)
        self.get_logger().info(f"VLM 瀹㈡埛绔凡鍚姩锛岃繛鎺ヨ嚦: {self.server_url}")

    def image_callback(self, msg):
        """鎺ユ敹鍥惧儚骞惰浆鎹负 OpenCV 鏍煎紡"""
        try:
            # ROS Image -> Numpy -> OpenCV
            np_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            # 娉ㄦ剰锛歊OS 閫氬父鏄?BGR锛孫penCV 涔熸槸 BGR锛屼絾 Qwen2-VL 闇€瑕?RGB
            # 濡傛灉浣犵殑 /camera/raw 鏄?BGR 鏍煎紡锛岃繖閲岄渶瑕佽浆鎹細
            self.latest_image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().warn(f"鍥惧儚瑙ｇ爜澶辫触: {e}")

    def planning_loop(self):
        """鍛ㄦ湡鎬у彂閫佽姹傜粰 Docker Server"""
        if self.latest_image is None:
            return 
            
        if self.is_requesting:
            return # 涓婁竴娆¤姹傝繕娌″洖鏉ワ紝璺宠繃杩欎竴甯?

        self.is_requesting = True
        
        try:
            # 1. 鍥惧儚缂栫爜 (Numpy -> Base64)
            # 浣跨敤 .jpg 鍘嬬缉鍙互鍑忓皯缃戠粶浼犺緭鑰楁椂锛屼笉褰卞搷绮惧害
            _, buffer = cv2.imencode('.jpg', self.latest_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # 2. 鏋勯€?Payload
            payload = {
                "image_base64": img_base64,
                "prompt": self.system_prompt
            }

            # 3. 鍙戦€?HTTP POST 璇锋眰 (璁剧疆瓒呮椂闃叉鍗℃)
            # 杩欏氨鏄綘鍦?test_vlm.py 閲屽仛鐨勪簨鎯?
            response = requests.post(self.server_url, json=payload, timeout=2.0)
            
            if response.status_code == 200:
                result_json = response.json()
                raw_text = result_json.get("result", "")
                
                # 4. 瑙ｆ瀽缁撴灉
                skill_id = self.parse_skill_id(raw_text)
                
                if skill_id is not None:
                    self.get_logger().info(f"鉁?VLM Response: '{raw_text}' -> Skill [{skill_id}]")
                    
                    # 鍙戝竷缁?Diffusion Policy
                    msg = Int8()
                    msg.data = skill_id
                    self.skill_pub.publish(msg)
                else:
                    self.get_logger().warn(f"鏃犳硶瑙ｆ瀽 Skill ID: {raw_text}")
            else:
                self.get_logger().error(f"Server 鎶ラ敊: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            self.get_logger().warn("VLM 璇锋眰瓒呮椂 (鍙兘鏄惧崱蹇?")
        except requests.exceptions.ConnectionError:
            self.get_logger().error("杩炴帴澶辫触! 璇锋鏌?Docker 瀹瑰櫒鏄惁鍚姩 (docker ps)")
        except Exception as e:
            self.get_logger().error(f"鏈煡閿欒: {e}")
        
        finally:
            self.is_requesting = False

    def parse_skill_id(self, text):
        """鎻愬彇鏁板瓧 0-3"""
        match = re.search(r'[0-3]', text)
        if match:
            return int(match.group())
        return None

def main():
    rclpy.init()
    node = VLMPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


