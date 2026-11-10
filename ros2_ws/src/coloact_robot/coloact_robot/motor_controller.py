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
        
        # 电机参数化电机控制器节点
        self.wheel_separation = 0.2
        self.wheel_radius = 0.05
        self.max_motor_speed = 255
        
        # 串口连接
        self.serial_port = None
        self.connect_serial()
        
        self.get_logger().info("电机控制器已启动")

    def connect_serial(self):
        """连接Arduino串口"""
        # 尝试常见的Arduino串口设备
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
                    time.sleep(2)  # 等待Arduino重启
                    self.get_logger().info(f"成功连接到Arduino: {port}")
                    break
                else:
                    self.get_logger().warning(f"串口不存在: {port}")
            except Exception as e:
                self.get_logger().warning(f"连接 {port} 失败: {e}")
        
        if self.serial_port is None:
            self.get_logger().error("无法连接到任何串口设备")

    def twist_to_motor_speeds(self, linear, angular):
        """将Twist消息转换为电机速度"""
        # 差速驱动模型
        left_speed = (linear - angular * self.wheel_separation / 2) / self.wheel_radius
        right_speed = (linear + angular * self.wheel_separation / 2) / self.wheel_radius
        
        # 转换为电机速度值
        left_motor = int((left_speed / 1.0) * self.max_motor_speed)
        right_motor = int((right_speed / 1.0) * self.max_motor_speed)
        
        # 限制范围
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
                
                # 可选：读取Arduino的响应
                # response = self.serial_port.readline().decode().strip()
                # if response:
                #     self.get_logger().info(f"Arduino响应: {response}")
                    
            except Exception as e:
                self.get_logger().error(f"发送命令失败: {e}")
                # 尝试重新连接
                self.connect_serial()
        else:
            self.get_logger().warning("串口未连接，无法发送命令")

    def cmd_callback(self, msg):
        """处理控制指令"""
        linear_speed = msg.linear.x
        angular_speed = msg.angular.z
        
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
        """析构函数，确保电机停止"""
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