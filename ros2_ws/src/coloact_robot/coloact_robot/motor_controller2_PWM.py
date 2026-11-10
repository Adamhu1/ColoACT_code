#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time
import os

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        
        # 订阅控制指令
        self.cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            10
        )
        
        # 参数配置
        self.wheel_separation = 0.2
        self.wheel_radius = 0.05
        self.max_motor_speed = 255  # PWM最大值
        self.max_linear_speed = 0.5  # 最大线速度 m/s
        self.max_angular_speed = 1.5  # 最大角速度 rad/s
        
        # 串口连接
        self.serial_port = None
        self.connect_serial()
        
        self.get_logger().info("电机控制器已启动 - PWM调速模式")

    def connect_serial(self):
        """连接Arduino串口"""
        possible_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
        
        for port in possible_ports:
            try:
                if os.path.exists(port):
                    self.serial_port = serial.Serial(
                        port=port,
                        baudrate=115200,
                        timeout=1,
                        write_timeout=1
                    )
                    time.sleep(2)
                    self.get_logger().info(f"成功连接到Arduino: {port}")
                    
                    # 等待READY消息
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        if self.serial_port.in_waiting > 0:
                            response = self.serial_port.readline().decode().strip()
                            self.get_logger().info(f"Arduino: {response}")
                            if "READY" in response:
                                break
                    break
            except Exception as e:
                self.get_logger().warning(f"连接 {port} 失败: {e}")
        
        if self.serial_port is None:
            self.get_logger().error("无法连接到任何串口设备")

    def twist_to_motor_speeds(self, linear, angular):
        """将Twist消息转换为电机速度"""
        # 差速驱动模型
        left_speed = (linear - angular * self.wheel_separation / 2) / self.wheel_radius
        right_speed = (linear + angular * self.wheel_separation / 2) / self.wheel_radius
        
        # 限制最大速度
        left_speed = max(min(left_speed, self.max_linear_speed), -self.max_linear_speed)
        right_speed = max(min(right_speed, self.max_linear_speed), -self.max_linear_speed)
        
        # 转换为电机PWM值（与速度成正比）
        left_motor = int((left_speed / self.max_linear_speed) * self.max_motor_speed)
        right_motor = int((right_speed / self.max_linear_speed) * self.max_motor_speed)
        
        # 限制PWM范围
        left_motor = max(min(left_motor, self.max_motor_speed), -self.max_motor_speed)
        right_motor = max(min(right_motor, self.max_motor_speed), -self.max_motor_speed)
        
        return left_motor, right_motor

    def send_to_arduino(self, left_speed, right_speed):
        """发送命令到Arduino"""
        if self.serial_port and self.serial_port.is_open:
            try:
                command = f"L{left_speed},R{right_speed}\n"
                self.serial_port.write(command.encode())
                self.get_logger().debug(f"发送命令: {command.strip()}")
                    
            except Exception as e:
                self.get_logger().error(f"发送命令失败: {e}")
                self.connect_serial()
        else:
            self.get_logger().warning("串口未连接，无法发送命令")

    def cmd_callback(self, msg):
        """处理控制指令"""
        linear_speed = msg.linear.x
        angular_speed = msg.angular.z
        
        # 限制输入速度范围
        linear_speed = max(min(linear_speed, self.max_linear_speed), -self.max_linear_speed)
        angular_speed = max(min(angular_speed, self.max_angular_speed), -self.max_angular_speed)
        
        # 转换为电机速度
        left_motor, right_motor = self.twist_to_motor_speeds(linear_speed, angular_speed)
        
        # 发送到Arduino
        self.send_to_arduino(left_motor, right_motor)
        
        # 记录日志
        self.get_logger().info(
            f"控制: 线速度={linear_speed:.2f}, 角速度={angular_speed:.2f} -> "
            f"左电机={left_motor}, 右电机={right_motor}"
        )

    def stop_motors(self):
        """停止电机"""
        self.send_to_arduino(0, 0)
        self.get_logger().info("停止所有电机")

    def __del__(self):
        if hasattr(self, 'serial_port') and self.serial_port:
            self.stop_motors()
            self.serial_port.close()

def main():
    rclpy.init()
    node = MotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_motors()
        node.get_logger().info("节点被用户中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()