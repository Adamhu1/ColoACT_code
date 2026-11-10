#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8, Bool

class JoySkillSender(Node):
    def __init__(self):
        super().__init__('joy_controller_skill_sender')
        
        # 1. 订阅手柄
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )
        
        # 2. 发布 Skill ID
        self.skill_pub = self.create_publisher(Int8, '/skill_id', 10)
        
        # 3. 发布 "是否允许AI运行" 的安全锁 (新增)
        self.enable_pub = self.create_publisher(Bool, '/autonomous_enabled', 10)
        
        self.current_skill = 0
        self.get_logger().info("🎮 手柄控制器 (安全版) 已启动")
        self.get_logger().info("⚠️  安全提示: 必须按住 [L1/LB] 键，机器人的一动才会生效！")
        self.get_logger().info("👉 按键映射: A=Skill0 | B=Skill1 | X=Skill2 | Y=Skill3")

    def joy_callback(self, msg):
        # === 1. 读取 Skill 选择 ===
        # Xbox/Logitech 常见映射: A=0, B=1, X=2, Y=3
        if msg.buttons[0] == 1: # A
            self.current_skill = 0
            self.get_logger().info(f"切换策略 -> Skill 0 (Go Straight)")
        elif msg.buttons[1] == 1: # B
            self.current_skill = 1
            self.get_logger().info(f"切换策略 -> Skill 1 (Turn Right)")
        elif msg.buttons[2] == 1: # X
            self.current_skill = 2
            self.get_logger().info(f"切换策略 -> Skill 2 (Turn Left)")
        elif msg.buttons[3] == 1: # Y
            self.current_skill = 3
            self.get_logger().info(f"切换策略 -> Skill 3 (Complex/Stop)")

        # === 2. 读取安全开关 (L1/LB) ===
        # 通常 L1 是 buttons[4] 或 buttons[5]，根据你的手柄调整
        # 如果你按别的键没反应，请尝试修改这个索引
        try:
            enable_ai = (msg.buttons[4] == 1) 
        except IndexError:
            enable_ai = False

        # === 3. 发布消息 ===
        # 发布 Skill
        skill_msg = Int8()
        skill_msg.data = self.current_skill
        self.skill_pub.publish(skill_msg)
        
        # 发布使能信号
        bool_msg = Bool()
        bool_msg.data = enable_ai
        self.enable_pub.publish(bool_msg)

def main(args=None):
    rclpy.init(args=args)
    node = JoySkillSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()