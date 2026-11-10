#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoyController(Node):  # 使用 JoyController 而不是 JoyControllerImproved
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
        
        # 手柄映射
        self.left_stick_vertical = 1    # 左摇杆上下
        self.left_stick_horizontal = 0  # 左摇杆左右
        
        # 改进的控制参数
        self.max_linear_speed = 0.3     # 降低最大线速度
        self.max_angular_speed = 1.0    # 降低最大角速度
        self.deadzone = 0.15            # 增大死区
        self.exponent = 1.5             # 指数曲线参数
        
        self.get_logger().info("改进版手柄控制器已启动 - 平滑速度映射")

    def apply_deadzone_and_curve(self, value):
        """应用死区和平滑曲线处理"""
        if abs(value) < self.deadzone:
            return 0.0
        
        sign = 1.0 if value >= 0 else -1.0
        abs_value = abs(value)
        
        # 归一化到 [0,1] 范围（减去死区）
        normalized = (abs_value - self.deadzone) / (1.0 - self.deadzone)
        
        # 应用指数曲线 - 让低速区域更细腻
        smoothed = sign * (normalized ** self.exponent)
        
        return smoothed

    # def joy_callback(self, msg):
    #     """处理手柄输入"""
    #     try:
    #         # 获取摇杆值并应用死区+平滑曲线
    #         linear_input = self.apply_deadzone_and_curve(msg.axes[self.left_stick_vertical])
    #         angular_input = self.apply_deadzone_and_curve(msg.axes[self.left_stick_horizontal])
            
    #         # 创建控制消息
    #         cmd_msg = Twist()
    #         cmd_msg.linear.x = -linear_input * self.max_linear_speed  # 前进后退
    #         cmd_msg.angular.z = -angular_input * self.max_angular_speed  # 转向，调整此处的正负号，可以调整手柄左右摇动与机器人左右运动的映射
            
    #         # 发布控制指令
    #         self.cmd_pub.publish(cmd_msg)
            
    #         # 调试信息（只在有显著输入时打印）
    #         if abs(linear_input) > 0.05 or abs(angular_input) > 0.05:
    #             self.get_logger().info(
    #                 f'改进版 - 摇杆: [{msg.axes[self.left_stick_vertical]:.2f}, {msg.axes[self.left_stick_horizontal]:.2f}] -> '
    #                 f'速度: [{cmd_msg.linear.x:.2f}, {cmd_msg.angular.z:.2f}]'
    #             )
            
    #     except Exception as e:
    #         self.get_logger().error(f"处理手柄数据错误: {e}")
    # def joy_callback(self, msg):
    #         """处理手柄输入 - 使用正确的轴映射"""
    #         try:
    #             # 获取摇杆值
    #             throttle_raw = msg.axes[self.left_stick_vertical]    # 轴1：上下
    #             steering_raw = msg.axes[self.left_stick_horizontal]  # 轴0：左右
                
    #             # 应用处理
    #             throttle = self.apply_deadzone_and_curve(throttle_raw)
    #             steering = self.apply_deadzone_and_curve(steering_raw)
                
    #             # 创建控制消息
    #             cmd_msg = Twist()
    #             cmd_msg.linear.x = -throttle * self.max_linear_speed   # 前进后退
    #             cmd_msg.angular.z = -steering * self.max_angular_speed  # 转向
                
    #             # 发布控制指令
    #             self.cmd_pub.publish(cmd_msg)
                
    #             # 运动状态判断和调试信息
    #             if abs(throttle) > 0.05 or abs(steering) > 0.05:
    #                 if throttle > 0.1 and steering > 0.1:
    #                     motion = "左前方弧线"  # 注意：steering正值是左转
    #                 elif throttle > 0.1 and steering < -0.1:
    #                     motion = "右前方弧线"  # steering负值是右转
    #                 elif throttle < -0.1 and steering > 0.1:
    #                     motion = "左后方弧线"
    #                 elif throttle < -0.1 and steering < -0.1:
    #                     motion = "右后方弧线"
    #                 elif abs(steering) > 0.1:
    #                     motion = "原地转向"
    #                 else:
    #                     motion = "直线运动"
                    
    #                 self.get_logger().info(
    #                     f"{motion}: 油门={throttle:.2f}, 转向={steering:.2f} | "
    #                     f"原始值: 上下轴={throttle_raw:.2f}, 左右轴={steering_raw:.2f}"
    #                 )
        
    #         except Exception as e:
    #             self.get_logger().error(f"处理手柄数据错误: {e}")


    def joy_callback(self, msg):
        """处理手柄输入 - 修正方向映射"""
        try:
            # 获取摇杆值
            throttle_raw = -msg.axes[self.left_stick_vertical]    # 轴1：上下
            steering_raw = msg.axes[self.left_stick_horizontal]  # 轴0：左右
            
            # 应用处理
            throttle = self.apply_deadzone_and_curve(throttle_raw)
            steering = self.apply_deadzone_and_curve(steering_raw)
            
            # 创建控制消息 - 确保正确映射
            cmd_msg = Twist()
            cmd_msg.linear.x = -throttle * self.max_linear_speed   # 前进后退
            cmd_msg.angular.z = -steering * self.max_angular_speed  # 转向（右转为正）
            
            # 发布控制指令
            self.cmd_pub.publish(cmd_msg)
            
            # 调试信息
            if abs(throttle) > 0.05 or abs(steering) > 0.05:
                direction = "前进" if throttle > 0 else "后退"
                turn = "右转" if steering > 0 else "左转" if steering < 0 else "直行"
                self.get_logger().info(f"{direction}+{turn}: 线速度={cmd_msg.linear.x:.2f}, 角速度={cmd_msg.angular.z:.2f}")
        
        except Exception as e:
            self.get_logger().error(f"处理手柄数据错误: {e}")

def main():
    rclpy.init()
    node = JoyController()  # 这里必须和类名一致！
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()