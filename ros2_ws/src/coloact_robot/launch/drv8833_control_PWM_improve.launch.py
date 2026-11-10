# ~/ros2_ws/src/coloact_robot/launch/coloact_robot_PWM.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 鍚姩鎵嬫焺椹卞姩鑺傜偣
        Node(
            package='joy_linux',
            executable='joy_linux_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',  # 鎵嬫焺璁惧锛屾牴鎹疄闄呮儏鍐典慨鏀?
                'deadzone': 0.1,
                'autorepeat_rate': 20.0
            }]
        ),
        
        # 鍚姩鎵嬫焺鎺у埗鍣ㄨ妭鐐?
        Node(
            package='coloact_robot',
            executable='joy_controller3_PWM',
            name='joy_controller3_PWM',
            output='screen'
        ),
        
        # 鍚姩鐢垫満鎺у埗鍣ㄨ妭鐐?
        Node(
            package='coloact_robot',
            executable='motor_controller3_PWM',
            name='motor_controller3_PWM',
            output='screen'
        )
    ])

