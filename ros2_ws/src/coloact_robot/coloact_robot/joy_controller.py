#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
import math

class JoyController(Node):
    def __init__(self):
        super().__init__('joy_controller')
        
        # 璁㈤槄鎵嬫焺璇濋
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )
        
        # 鍙戝竷鎺у埗鎸囦护
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 鎵嬫焺鏄犲皠 (鏍规嵁浣犵殑鎵嬫焺璋冩暣)
        self.left_stick_vertical = 1    # 宸︽憞鏉嗕笂涓?
        self.left_stick_horizontal = 0  # 宸︽憞鏉嗗乏鍙?
        self.right_stick_vertical = 4   # 鍙虫憞鏉嗕笂涓?
        self.right_stick_horizontal = 3 # 鍙虫憞鏉嗗乏鍙?
        
        # 鎺у埗鍙傛暟
        self.max_linear_speed = 1.0     # 鏈€澶х嚎閫熷害 m/s
        self.max_angular_speed = 2.0    # 鏈€澶ц閫熷害 rad/s
        self.deadzone = 0.1             # 姝诲尯闃堝€?
        
        self.get_logger().info("鎵嬫焺鎺у埗鍣ㄥ凡鍚姩")

    def apply_deadzone(self, value):
        """搴旂敤姝诲尯澶勭悊"""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def joy_callback(self, msg):
        """澶勭悊鎵嬫焺杈撳叆"""
        try:
            # 鑾峰彇鎽囨潌鍊煎苟搴旂敤姝诲尯
            linear_x = self.apply_deadzone(msg.axes[self.left_stick_vertical])
            angular_z = self.apply_deadzone(msg.axes[self.left_stick_horizontal])
            
            # 鍒涘缓鎺у埗娑堟伅
            cmd_msg = Twist()
            cmd_msg.linear.x = -linear_x * self.max_linear_speed  # 鍓嶈繘鍚庨€€
            cmd_msg.angular.z = angular_z * self.max_angular_speed  # 杞悜
            
            # 鍙戝竷鎺у埗鎸囦护
            self.cmd_pub.publish(cmd_msg)
            
            # 鎵撳嵃璋冭瘯淇℃伅
            self.get_logger().debug(
                f'绾块€熷害: {cmd_msg.linear.x:.2f}, 瑙掗€熷害: {cmd_msg.angular.z:.2f}'
            )
            
        except Exception as e:
            self.get_logger().error(f"澶勭悊鎵嬫焺鏁版嵁閿欒: {e}")

def main():
    rclpy.init()
    node = JoyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


