from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 纭欢椹卞姩: 鐢垫満
        Node(
            package='coloact_robot',
            executable='motor_controller3_PWM_yibianman',
            name='motor_driver',
            output='screen'
        ),  # <--- 杩欓噷鐨勯€楀彿闈炲父閲嶈锛?
        
        # 2. 纭欢椹卞姩: 鎵嬫焺 (joy_linux)
        # 浣跨敤浣犱箣鍓嶉獙璇佹垚鍔熺殑 joy_linux 鏂规
        Node(
            package='joy_linux', 
            executable='joy_linux_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',
                'deadzone': 0.05,
                'autorepeat_rate': 20.0
            }]
        ),  # <--- 杩欓噷鐨勯€楀彿涔熶笉鑳藉皯锛?

        # 3. 鎵嬫焺閫昏緫鑺傜偣 (Switch Skill + Manual Control)
        # 浣跨敤鏂板缓鐨?v4 鐗堟湰
        Node(
            package='coloact_robot',
            executable='joy_controller3_for_inference_PWM', 
            name='joy_controller',
            output='screen'
        ),  # <--- 閫楀彿

        # 4. 鎰熺煡灞?(Depth Anything)
        Node(
            package='coloact_robot',
            executable='depth_processing_node', 
            name='depth_processor',
            parameters=[{'camera_id': 0}, {'processing_size': [320, 240]}]
        ),  # <--- 閫楀彿
        
        # 5. 鎺у埗灞?(Diffusion Policy)
        Node(
            package='coloact_robot',
            executable='diffusion_inference_node',
            name='diffusion_pilot',
            output='screen'
        )
    ])

