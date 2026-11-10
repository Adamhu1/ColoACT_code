from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 鍚姩鐢垫満椹卞姩 (搴曞眰鎵ц)
        Node(
            package='coloact_robot',
            executable='motor_controller3_PWM_yibianman',
            name='motor_driver',
            output='screen'
        ),
        
        # 2. 鍚姩娣卞害鎰熺煡 (瑙嗚杈撳叆)
        Node(
            package='coloact_robot',
            executable='depth_processing_node', 
            name='depth_processor',
            parameters=[
                {'camera_id': 0},
                {'processing_size': [320, 240]} # 蹇呴』涓庤缁冩椂涓€鑷?
            ]
        ),
        
        # 3. 鍚姩 BC 妯′豢瀛︿範鎺ㄧ悊鑺傜偣 (澶ц剳)
        Node(
            package='coloact_robot',
            executable='bc_inference_node',
            name='bc_pilot',
            output='screen'
        )
    ])

