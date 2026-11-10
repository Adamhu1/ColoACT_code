from launch import LaunchDescription
from launch_ros.actions import Node

#鐢ㄤ簬鍘熷瓙鍖栦换鍔″悇鑺傜偣鍚姩鐨刲aunch鏂囦欢

def generate_launch_description():
    return LaunchDescription([
        # 1. 鍚姩鎵嬫焺椹卞姩 (纭欢灞?
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
        
        # 2. [淇敼] 鍚姩甯︽爣绛剧殑鎵嬫焺鎺у埗鍣?(閫昏緫灞?
        Node(
            package='coloact_robot',
            executable='joy_controller_labeled', # 纭繚鍦?setup.py 涓敞鍐屼簡
            name='joy_controller',
            output='screen'
        ),
        
        # 3. 鍚姩鐢垫満椹卞姩 (鎵ц灞?
        Node(
            package='coloact_robot',
            executable='motor_controller3_PWM_yibianman',
            name='motor_driver',
            output='screen'
        ),
        
        # 4. 鍚姩 Depth Anything V2 鑺傜偣
        Node(
            package='coloact_robot',
            executable='depth_processing_node', 
            name='depth_processor',
            parameters=[
                {'camera_id': 0},
                {'debug_mode': False},
                {'processing_size': [320, 240]} 
            ]
        ),
        
        # 5. [淇敼] 鍚姩甯︽爣绛剧殑鏁版嵁閲囬泦鑺傜偣
        Node(
            package='coloact_robot',
            executable='data_collector_labeled', # 纭繚鍦?setup.py 涓敞鍐屼簡
            name='data_collector',
            output='screen'
        )
    ])

