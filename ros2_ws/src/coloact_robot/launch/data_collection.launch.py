from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 鍚姩鎵嬫焺椹卞姩 (璇诲彇纭欢 /dev/input/js0)
        Node(
            package='joy_linux',
            executable='joy_linux_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',
                'deadzone': 0.05,
                'autorepeat_rate': 20.0
            }]
        ),
        
        # 2. 鍚姩鎵嬫焺鎺у埗鍣?(灏嗘墜鏌勪俊鍙疯浆鎹负 cmd_vel)
        Node(
            package='coloact_robot',
            executable='joy_controller3_PWM',
            name='joy_controller',
            output='screen'
        ),
        
        # 3. 鍚姩鐢垫満椹卞姩 (灏?cmd_vel 鍙戦€佺粰 Arduino)
        Node(
            package='coloact_robot',
            executable='motor_controller3_PWM_yibianman',
            name='motor_driver',
            output='screen'
        ),
        
        # 4. 鍚姩 Depth Anything V2 娣卞害/楂樼▼浼拌鑺傜偣
        # 娉ㄦ剰锛氳纭繚浣犲湪 setup.py 涓皢 w_danmu_... 娉ㄥ唽涓轰簡 depth_estimator
        Node(
            package='coloact_robot',
            executable='depth_processing_node', 
            name='depth_processor',
            parameters=[
                {'camera_id': 0},
                {'debug_mode': False},
                # 寮哄埗涓婃父杈撳嚭 320x240锛岃繖鏍?data_collector 閲岀殑缂╂斁寰幆灏变笉浼氳瑙﹀彂
                {'processing_size': [320, 240]} 
            ]
        ),
        
        # 5. 鍚姩鏁版嵁閲囬泦鑺傜偣
        Node(
            package='coloact_robot',
            executable='data_collector_direct',
            name='data_collector',
            output='screen' # 灏嗘棩蹇楄緭鍑哄埌灞忓箷锛屾柟渚跨湅褰曞埗鐘舵€?
        )
    ])

