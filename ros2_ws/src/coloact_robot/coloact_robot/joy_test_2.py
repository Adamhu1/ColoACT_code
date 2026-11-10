#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import time

class SimpleJoyTest(Node):
    def __init__(self):
        super().__init__('simple_joy_test')
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.listener_callback,
            10)
        self.last_time = time.time()
        
    def listener_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_time > 1.0:  # 姣忕鎵撳嵃涓€娆?
            self.last_time = current_time
            print("Axes:", [f"{x:.2f}" for x in msg.axes])
            print("Buttons:", msg.buttons)
            print("---")

def main():
    rclpy.init()
    node = SimpleJoyTest()
    print("绛夊緟鎵嬫焺鏁版嵁...")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


