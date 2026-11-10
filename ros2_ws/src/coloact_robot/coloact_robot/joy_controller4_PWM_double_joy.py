#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoyController(Node):
    def __init__(self):
        super().__init__('joy_controller_improved')
        
        # 订阅手柄话题
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )
        
        # 发布控制指令
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 手柄映射 - 新的映射方案
        self.left_stick_vertical = 1    # 左摇杆上下 → 油门（前进/后退）
        self.right_stick_horizontal = 3 # 右摇杆左右 → 转向差速
        
        # 控制参数
        self.max_linear_speed = 0.4     # 最大线速度 m/s
        self.max_angular_speed = 1.5    # 最大角速度 rad/s
        self.deadzone = 0.15            # 死区阈值
        self.exponent = 2            # 平滑曲线参数
        
        self.get_logger().info("改进版手柄控制器已启动 - 左摇杆油门，右摇杆转向")

    def apply_deadzone_and_curve(self, value):
        """应用死区和平滑曲线处理"""
        if abs(value) < self.deadzone:
            return 0.0
        
        sign = 1.0 if value >= 0 else -1.0
        abs_value = abs(value)
        
        # 归一化到 [0,1] 范围（减去死区）
        normalized = (abs_value - self.deadzone) / (1.0 - self.deadzone)
        
        # 应用指数曲线
        smoothed = sign * (abs_value ** self.exponent)
        return smoothed

    def joy_callback(self, msg):
        """处理手柄输入 - 新控制方案"""
        try:
            # 左摇杆：控制油门（前进/后退）
            throttle = self.apply_deadzone_and_curve(msg.axes[self.left_stick_vertical])
            
            # 右摇杆：控制转向差速
            steering = self.apply_deadzone_and_curve(msg.axes[self.right_stick_horizontal])
            
            # 创建控制消息
            cmd_msg = Twist()
            cmd_msg.linear.x = -throttle * self.max_linear_speed   # 前进后退
            cmd_msg.angular.z = -steering * self.max_angular_speed  # 转向
            
            # 发布控制指令
            self.cmd_pub.publish(cmd_msg)
            
            # 调试信息（只在有显著输入时打印）
            if abs(throttle) > 0.05 or abs(steering) > 0.05:
                self.get_logger().info(
                    f'控制: 油门={-throttle:.2f}, 转向={-steering:.2f} -> '
                    f'线速度={cmd_msg.linear.x:.2f}, 角速度={cmd_msg.angular.z:.2f}'
                )
            
        except Exception as e:
            self.get_logger().error(f"处理手柄数据错误: {e}")

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