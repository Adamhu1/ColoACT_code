#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time
import os
#相比于motor_controller3_PWM copy.py，这个方案是转动时有一边电机转速为0的方案
class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        # 参数配置
        self.wheel_separation = 0.014 #轮距 m
        self.wheel_radius = 0.0035 # 轮子半径 m
        self.max_motor_speed = 255  # PWM最大值
        self.max_linear_speed = 0.3  # 最大线速度 m/s
        self.max_angular_speed = 0.8  # 最大角速度 rad/s

        # 添加速度缩放因子
        self.speed_scale = 0.8  # 整体速度缩放

         # 串口连接
        self.serial_port = None
        self.connect_serial()

        self.get_logger().info("电机控制器已启动 - 改进的速度控制")


        
        

        

        # 订阅控制指令
        self.cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            10
        )
        
        
       
        
        

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



    # def twist_to_motor_speeds(self, linear, angular):
    #     """改进的差速驱动：转向时两侧电机都向前但速度不同"""
    #     # 基础速度（基于线速度）
    #     base_speed = linear / self.wheel_radius
    #     base_motor = int((base_speed / self.max_linear_speed) * self.max_motor_speed)
    #     base_motor = max(min(base_motor, self.max_motor_speed), -self.max_motor_speed)
        
    #     # 转向修正量（基于角速度）
    #     turn_correction = int((abs(angular) / self.max_angular_speed) * self.max_motor_speed * 0.5)
        
    #     if angular > 0.05:  # 右转
    #         # 左电机更快，右电机较慢
    #         left_motor = base_motor + turn_correction
    #         right_motor = base_motor - turn_correction
    #         turn_type = f"右转(左快右慢)"
    #     elif angular < -0.05:  # 左转
    #         # 右电机更快，左电机较慢
    #         left_motor = base_motor - turn_correction
    #         right_motor = base_motor + turn_correction
    #         turn_type = f"左转(右快左慢)"
    #     else:  # 直行
    #         left_motor = base_motor
    #         right_motor = base_motor
    #         turn_type = "直行"
        
    #     # 确保两电机都是向前（同向）
    #     # 如果基础速度是向前，确保转向后两电机都 >= 0
    #     if base_motor >= 0:
    #         left_motor = max(0, left_motor)
    #         right_motor = max(0, right_motor)
    #     else:  # 基础速度是向后
    #         left_motor = min(0, left_motor)
    #         right_motor = min(0, right_motor)
        
    #     # 限制范围
    #     max_single_motor = int(self.max_motor_speed * 0.9)
    #     left_motor = max(min(left_motor, max_single_motor), -max_single_motor)
    #     right_motor = max(min(right_motor, max_single_motor), -max_single_motor)
        
    #     self.get_logger().info(f"差速模式: {turn_type} | 左={left_motor}, 右={right_motor}")
        
    #     return left_motor, right_motor


    # def twist_to_motor_speeds(self, linear, angular):
    #     """改进的差速驱动：确保两电机同向转动，通过速度差实现转向"""
        
    #     # 基础线速度转换
    #     base_speed = linear / self.wheel_radius
    #     base_motor = int((base_speed / self.max_linear_speed) * self.max_motor_speed)
    #     base_motor = max(min(base_motor, self.max_motor_speed), -self.max_motor_speed)
        
    #     # 应用速度缩放
    #     base_motor = int(base_motor * self.speed_scale)
        
    #     # 转向强度 (0到1)
    #     turn_strength = min(abs(angular) / self.max_angular_speed, 1.0)
        
    #     # 改进的转向逻辑：确保两电机同向
    #     if abs(angular) > 0.05:  # 有转向输入
    #         # 基础速度差计算 - 增加转弯灵敏度
    #         base_turn_diff = int(self.max_motor_speed * 0.6 * turn_strength)  # 增加基础差值
            
    #         if angular > 0.05:  # 右转 - 左电机更快，右电机较慢
    #             left_motor = base_motor + base_turn_diff
    #             right_motor = base_motor - base_turn_diff
    #             turn_info = f"右转(左+{base_turn_diff})"
    #         else:  # 左转 - 右电机更快，左电机较慢  
    #             left_motor = base_motor - base_turn_diff
    #             right_motor = base_motor + base_turn_diff
    #             turn_info = f"左转(右+{base_turn_diff})"
            
    #         # 确保两电机同向（不出现一正一负）
    #         if base_motor >= 0:  # 前进或停止
    #             left_motor = max(0, left_motor)
    #             right_motor = max(0, right_motor)
    #         else:  # 后退
    #             left_motor = min(0, left_motor)  
    #             right_motor = min(0, right_motor)
                
    #     else:  # 直行
    #         left_motor = base_motor
    #         right_motor = base_motor
    #         turn_info = "直行"
        
    #     # 限制范围并确保最小转弯速度
    #     max_single_motor = int(self.max_motor_speed * 0.95)  # 提高上限
        
    #     # 如果有转向但速度差太小，强制增加最小速度差
    #     if abs(angular) > 0.1 and abs(left_motor - right_motor) < 40:
    #         min_diff = 60  # 最小速度差
    #         if angular > 0:  # 右转
    #             left_motor = max(left_motor, min_diff) if base_motor >= 0 else min(left_motor, -min_diff)
    #             right_motor = max(0, right_motor) if base_motor >= 0 else min(0, right_motor)
    #         else:  # 左转
    #             right_motor = max(right_motor, min_diff) if base_motor >= 0 else min(right_motor, -min_diff)  
    #             left_motor = max(0, left_motor) if base_motor >= 0 else min(0, left_motor)
        
    #     left_motor = max(min(left_motor, max_single_motor), -max_single_motor)
    #     right_motor = max(min(right_motor, max_single_motor), -max_single_motor)
        
    #     # 调试信息
    #     if abs(angular) > 0.05:
    #         self.get_logger().info(f"🔄 {turn_info} | 基础={base_motor}, 左={left_motor}, 右={right_motor}, 差值={abs(left_motor-right_motor)}")
        
    #     return left_motor, right_motor

    # def twist_to_motor_speeds(self, linear, angular):
    #     """全新的差速驱动逻辑：确保两电机都转动，通过速度差实现转向"""
        
    #     # 基础线速度转换
    #     base_speed = linear / self.wheel_radius
    #     base_motor = int((base_speed / self.max_linear_speed) * self.max_motor_speed)
    #     base_motor = max(min(base_motor, self.max_motor_speed), -self.max_motor_speed)
        
    #     # 应用速度缩放
    #     base_motor = int(base_motor * self.speed_scale)
        
    #     # 转向强度 (0到1)
    #     turn_strength = min(abs(angular) / self.max_angular_speed, 1.0)
        
    #     # 全新的转向逻辑
    #     if abs(angular) > 0.05:  # 有转向输入
    #         # 计算转向速度差 - 基于基础速度和转向强度的组合
    #         if abs(base_motor) < 20:  # 基础速度很小时
    #             # 确保有最小运动速度
    #             min_speed = 80  # 最小PWM值确保电机转动
    #             turn_diff = int(self.max_motor_speed * 0.4 * turn_strength)  # 转向差值
                
    #             if angular > 0.05:  # 右转
    #                 left_motor = min_speed + turn_diff
    #                 right_motor = min_speed - turn_diff
    #                 turn_info = f"低速右转(左{left_motor}, 右{right_motor})"
    #             else:  # 左转
    #                 left_motor = min_speed - turn_diff
    #                 right_motor = min_speed + turn_diff
    #                 turn_info = f"低速左转(左{left_motor}, 右{right_motor})"
                    
    #             # 确保方向正确
    #             if base_motor < 0:  # 如果是后退
    #                 left_motor = -left_motor
    #                 right_motor = -right_motor
                    
    #         else:  # 有显著的基础速度
    #             # 基于基础速度计算转向差值
    #             turn_diff = int(abs(base_motor) * 0.8 * turn_strength)  # 增加转向比例
                
    #             if angular > 0.05:  # 右转
    #                 left_motor = base_motor + turn_diff
    #                 right_motor = base_motor - turn_diff
    #                 turn_info = f"右转(左+{turn_diff})"
    #             else:  # 左转
    #                 left_motor = base_motor - turn_diff
    #                 right_motor = base_motor + turn_diff
    #                 turn_info = f"左转(右+{turn_diff})"
                
    #         # 强制确保两电机都转动（除非完全停止）
    #         min_active_speed = 30  # 最小激活速度
    #         if base_motor > 0:  # 前进
    #             left_motor = max(min_active_speed, left_motor)
    #             right_motor = max(min_active_speed, right_motor)
    #         elif base_motor < 0:  # 后退
    #             left_motor = min(-min_active_speed, left_motor)
    #             right_motor = min(-min_active_speed, right_motor)
                
    #     else:  # 直行
    #         left_motor = base_motor
    #         right_motor = base_motor
    #         turn_info = "直行"
        
    #     # 限制范围
    #     max_single_motor = int(self.max_motor_speed * 0.95)
    #     left_motor = max(min(left_motor, max_single_motor), -max_single_motor)
    #     right_motor = max(min(right_motor, max_single_motor), -max_single_motor)
        
    #     # 最终检查：如果转向时一个电机为0，强制给最小速度
    #     if abs(angular) > 0.1:
    #         if left_motor == 0 and base_motor >= 0:
    #             left_motor = 50
    #         elif right_motor == 0 and base_motor >= 0:
    #             right_motor = 50
    #         elif left_motor == 0 and base_motor < 0:
    #             left_motor = -50
    #         elif right_motor == 0 and base_motor < 0:
    #             right_motor = -50
        
    #     # 详细的调试信息
    #     if abs(angular) > 0.05:
    #         self.get_logger().info(f"🔄 {turn_info} | 基础={base_motor}, 左={left_motor}, 右={right_motor}")
    #         self.get_logger().info(f"   转向输入: 角速度={angular:.2f}, 转向强度={turn_strength:.2f}")
        
    #     return left_motor, right_motor


    def twist_to_motor_speeds(self, linear, angular):
        """简化的差速驱动：确保两电机都转动"""
        
        # 基础速度计算
        base_motor = int((linear / self.max_linear_speed) * self.max_motor_speed)
        base_motor = max(min(base_motor, self.max_motor_speed), -self.max_motor_speed)
        
        # 应用速度缩放
        base_motor = int(base_motor * self.speed_scale)
        
        # 转向处理
        if abs(angular) > 0.05:  # 有转向输入
            turn_strength = min(abs(angular) / self.max_angular_speed, 1.0)
            
            # 计算速度差 - 基于最大速度和转向强度
            speed_diff = int(self.max_motor_speed * 0.5 * turn_strength)
            
            # 确保有最小基础速度用于转向
            if abs(base_motor) < 30:  # 基础速度太小
                min_base_speed = 60  # 最小基础速度
                base_motor = min_base_speed if base_motor >= 0 else -min_base_speed
            
            if angular > 0:  # 右转 - 左电机更快
                left_motor = base_motor + speed_diff
                right_motor = base_motor - speed_diff
                turn_info = f"右转(左快)"
            else:  # 左转 - 右电机更快
                left_motor = base_motor - speed_diff
                right_motor = base_motor + speed_diff
                turn_info = f"左转(右快)"
            
            # 强制确保两电机同向且都转动
            min_active_speed = 40
            if base_motor > 0:  # 前进
                left_motor = max(min_active_speed, left_motor)
                right_motor = max(min_active_speed, right_motor)
            else:  # 后退
                left_motor = min(-min_active_speed, left_motor)
                right_motor = min(-min_active_speed, right_motor)
                
        else:  # 直行
            left_motor = base_motor
            right_motor = base_motor
            turn_info = "直行"
        
        # 限制范围
        max_val = 200
        left_motor = max(min(left_motor, max_val), -max_val)
        right_motor = max(min(right_motor, max_val), -max_val)
        
        # 调试信息
        if abs(angular) > 0.05:
            self.get_logger().info(f"🔄 {turn_info} | 基础={base_motor}, 左={left_motor}, 右={right_motor}, 差值={abs(left_motor-right_motor)}")
        
        return left_motor, right_motor

    def send_to_arduino(self, left_speed, right_speed):
        """发送命令到Arduino - 修正交换问题"""
        if self.serial_port and self.serial_port.is_open:
            try:
                # 修正：不要交换左右电机！
                command = f"L{left_speed},R{right_speed}\n"  # 正常顺序
                self.serial_port.write(command.encode())
                self.get_logger().debug(f"发送命令: {command.strip()}")
                    
            except Exception as e:
                self.get_logger().error(f"发送命令失败: {e}")
                self.connect_serial()
        else:
            self.get_logger().warning("串口未连接，无法发送命令")

    def send_to_arduino(self, left_speed, right_speed):
        """发送命令到Arduino"""
        if self.serial_port and self.serial_port.is_open:
            try:
                # 如果方向还是不对，尝试交换左右电机
                command = f"L{right_speed},R{left_speed}\n"  # 交换左右
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

        self.get_logger().info(f"电机PWM: 左={left_motor}, 右={right_motor}")

        # 检查运动方向
        if linear_speed > 0 and angular_speed > 0:
            self.get_logger().info("运动: 前进 + 右转")
        elif linear_speed > 0 and angular_speed < 0:
            self.get_logger().info("运动: 前进 + 左转") 
        elif linear_speed < 0 and angular_speed > 0:
            self.get_logger().info("运动: 后退 + 右转 ← 这就是问题！")
        elif linear_speed < 0 and angular_speed < 0:
            self.get_logger().info("运动: 后退 + 左转")
        
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