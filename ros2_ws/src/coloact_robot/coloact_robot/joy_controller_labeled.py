#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8

class JoyControllerLabeled(Node):
    def __init__(self):
        super().__init__('joy_controller_labeled')
        
        # 璁㈤槄鎵嬫焺
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        
        # 鍙戝竷鎺у埗鎸囦护
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 鍙戝竷 Skill ID
        self.skill_pub = self.create_publisher(Int8, 'skill_id', 10)
        
        # 鎽囨潌鏄犲皠 (鏍规嵁浣犵殑鍘熸枃浠朵繚鐣欓厤缃?
        self.left_stick_vertical = 1
        self.left_stick_horizontal = 0
        
        # 鎸夐敭鏄犲皠 (4涓師瀛愪换鍔?
        # 0:A (鐩磋), 1:B (瑙傚療/鎮仠), 2:X (杩囧集), 3:Y (鑴卞洶)
        self.btn_map = {0: 0, 1: 1, 2: 2, 3: 3}
        
        self.current_skill = 0 
        
        # 閫熷害鍙傛暟
        self.max_linear_speed = 0.3
        self.max_angular_speed = 1.0
        self.deadzone = 0.15
        self.exponent = 1.5

        self.get_logger().info("甯︽爣绛剧殑鎵嬫焺鎺у埗鍣ㄥ凡鍚姩 (淇濈暀鍘熻礋鍙烽€昏緫)")

    def apply_deadzone_and_curve(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        normalized = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return sign * (normalized ** self.exponent)

    def joy_callback(self, msg):
        try:
            # --- 1. 鍒囨崲 Skill ID ---
            for btn_idx, skill_id in self.btn_map.items():
                if btn_idx < len(msg.buttons) and msg.buttons[btn_idx]:
                    self.current_skill = skill_id
                    # 鍙€夛細鎵撳嵃鍒囨崲鏃ュ織
                    # self.get_logger().info(f"鍒囨崲妯″紡 -> Skill {skill_id}")
            
            # 鍙戝竷褰撳墠 Skill ID
            skill_msg = Int8()
            skill_msg.data = self.current_skill
            self.skill_pub.publish(skill_msg)

            # --- 2. 澶勭悊杩愬姩杈撳叆 (淇濈暀浣犱範鎯殑鍘熷閫昏緫) ---
            # 杩欓噷涓嶅姞璐熷彿锛屼繚鎸佸師濮嬭鍙?
            throttle_raw = -msg.axes[self.left_stick_vertical]
            steering_raw = msg.axes[self.left_stick_horizontal]
            
            throttle = self.apply_deadzone_and_curve(throttle_raw)
            steering = self.apply_deadzone_and_curve(steering_raw)
            
            cmd_msg = Twist()
            
            # --- 3. 鏍规嵁 4绉?Skill 璋冩暣鎺у埗鍙傛暟 ---
            
            if self.current_skill == 1: 
                # Skill 1: [瑙傚療/鎮仠] (鎸?B 閿?
                # 鏋佹參閫燂紝鏂逛究寰皟瀵瑰噯鐥呯伓
                scale = 0.2
                # 浣跨敤浣犺姹傜殑 -throttle 褰㈠紡
                cmd_msg.linear.x = -throttle * self.max_linear_speed * scale
                cmd_msg.angular.z = -steering * self.max_angular_speed * scale
                
            elif self.current_skill == 2: 
                # Skill 2: [杩囧集] (鎸?X 閿?
                # 閫熷害鍑忓崐锛岃浆鍚戠伒鏁忓害杈冮珮(0.8)锛岄€傚悎S寮?
                cmd_msg.linear.x = -throttle * self.max_linear_speed * 0.5
                cmd_msg.angular.z = -steering * self.max_angular_speed * 0.8
                
            elif self.current_skill == 3: 
                # Skill 3: [鑴卞洶] (鎸?Y 閿?
                # 閫熷害鍑忓崐锛屽叏鍔熺巼杞悜锛岄€傚悎鍦ㄦ鑳″悓璋冩暣
                cmd_msg.linear.x = -throttle * self.max_linear_speed * 0.5
                cmd_msg.angular.z = -steering * self.max_angular_speed * 1.0
                
            else: 
                # Skill 0: [鐩磋/閫氱敤] (鎸?A 閿?
                # 鍏ㄩ€燂紝鏍囧噯杞悜
                cmd_msg.linear.x = -throttle * self.max_linear_speed
                cmd_msg.angular.z = -steering * self.max_angular_speed
            
            # 鍙戝竷鎸囦护
            self.cmd_pub.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f"鎵嬫焺澶勭悊閿欒: {e}")

def main():
    rclpy.init()
    node = JoyControllerLabeled()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


