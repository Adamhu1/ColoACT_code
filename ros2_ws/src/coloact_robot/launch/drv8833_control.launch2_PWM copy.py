from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 启动手柄驱动节点
        Node(
            package='joy_linux',
            executable='joy_linux_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',  # 手柄设备，根据实际情况修改
                'deadzone': 0.1,
                'autorepeat_rate': 20.0
            }]
        ),
        
        # 启动手柄控制器节点
        Node(
            package='coloact_robot',
            executable='joy_controller',
            name='joy_controller',
            output='screen'
        ),
        
        # 启动电机控制器节点
        Node(
            package='coloact_robot',
            executable='motor_controller2_PWM',
            name='motor_controller2_PWM',
            output='screen'
        )
    ])