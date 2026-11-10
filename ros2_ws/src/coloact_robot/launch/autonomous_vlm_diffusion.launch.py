from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 纭欢椹卞姩
        Node(package='coloact_robot', executable='motor_controller3_PWM_yibianman', name='motor_driver'),
        
        # 2. 鎰熺煡灞?(Depth Anything V2)
        Node(package='coloact_robot', executable='depth_processing_node', name='depth_processor',
             parameters=[{'camera_id': 0}, {'processing_size': [320, 240]}]),
        
        # 3. 瑙勫垝灞?(VLM - 鏂板)
        Node(package='coloact_robot', executable='vlm_planner_node', name='vlm_brain', output='screen'),
        
        # 4. 鎺у埗灞?(Diffusion Policy)
        # 娉ㄦ剰锛氱幇鍦ㄤ笉闇€瑕佹墜鏌勪簡锛屽畠鍚?VLM 鐨?/skill_id
        Node(package='coloact_robot', executable='diffusion_inference_node', name='diffusion_pilot', output='screen')
    ])

