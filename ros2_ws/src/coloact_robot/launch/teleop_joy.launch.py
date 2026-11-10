from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # joy鑺傜偣
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen'
    )
    
    # teleop_twist_joy鑺傜偣 - 鐩存帴浼犻€掑弬鏁?
    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        output='screen',
        parameters=[{
            'axis_linear.x': 1,
            'axis_angular.yaw': 0,
            'scale_linear.x': 0.4,
            'scale_angular.yaw': 0.8,
            'enable_button': -1,
            'require_enable_button': False,
            'axis_deadzone': 0.1
        }]
    )
    
    # 鐢垫満鎺у埗鍣?
    motor_controller_node = Node(
        package='coloact_robot',
        executable='motor_controller3_PWM',
        name='motor_controller',
        output='screen'
    )
    
    return LaunchDescription([
        joy_node,
        teleop_node,
        motor_controller_node,
    ])

