#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 澹版槑鍙傛暟
    return LaunchDescription([
        # 鍙傛暟澹版槑
        DeclareLaunchArgument(
            'joystick_device',
            default_value='/dev/input/js0',
            description='鎵嬫焺璁惧璺緞'
        ),

        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/ttyACM0',
            description='Arduino涓插彛璺緞'
        ),

        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='鎽勫儚澶碔D'
        ),

        DeclareLaunchArgument(
            'debug_mode',
            default_value='false',
            description='鏄惁鍚敤璋冭瘯妯″紡'
        ),

        DeclareLaunchArgument(
            'enable_control_output',
            default_value='false',
            description='鏄惁鍚敤鎺у埗杈撳嚭锛堟繁搴﹁妭鐐规帶鍒剁數鏈猴級'
        ),

        # 鎵嬫焺椹卞姩鑺傜偣
        Node(
            package='joy_linux',
            executable='joy_linux_node',
            name='joy_node',
            parameters=[{
                'dev': LaunchConfiguration('joystick_device'),
                'deadzone': 0.1,
                'autorepeat_rate': 20.0
            }]
        ),

        # 鎵嬫焺鎺у埗鑺傜偣
        Node(
            package='coloact_robot',
            executable='joy_controller3_PWM',
            name='joy_controller3_PWM',
            output='screen',
            parameters=[{
                'max_linear_speed': 1.0,
                'max_angular_speed': 2.0
            }]
        ),

        # 鐢垫満鎺у埗鍣ㄨ妭鐐?
        Node(
            package='coloact_robot',
            executable='motor_controller3_PWM_yibianman',
            name='motor_controller3_PWM_yibianman',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'baud_rate': 115200
            }]
        ),

        # 娣卞害澶勭悊鑺傜偣 (闆嗘垚鍒癲rv8833_control鍖呬腑)
        Node(
            package='coloact_robot',
            executable='depth_processing_node',
            name='depth_processing_node',
            output='screen',
            parameters=[{
                'camera_id': LaunchConfiguration('camera_id'),
                'processing_size': [640, 480],
                'processing_frequency': 10.0,
                'depth_threshold': 1000,
                'debug_mode': LaunchConfiguration('debug_mode'),
                'enable_control_output': LaunchConfiguration('enable_control_output')
            }]
        ),
    ])

