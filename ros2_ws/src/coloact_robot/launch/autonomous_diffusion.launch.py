from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # ================= 閰嶇疆鍖哄煙 =================
    # 鍦ㄨ繖閲屽垏鎹綘瑕佹祴璇曠殑妯″紡
    INPUT_MODE = 'rgbd_elev'  # 鍙€? 'rgb', 'rgbd', 'rgbd_elev'
    
    # 鏄惁寮€鍚畨鍏ㄩ攣?
    # True = 蹇呴』鎸夋墜鏌凩1鎵嶈蛋; False = 鍚姩鍗宠蛋(鍗遍櫓!)
    AUTO_START = False 
    
    # 浣犵殑妯″瀷璺緞 (濡傛灉鐣欑┖锛屼細浣跨敤浠ｇ爜閲岄粯璁ょ殑璺緞)
    # 寤鸿濉叆缁濆璺緞锛岀‘淇濆姞杞芥纭?
    MODEL_PATH = '/path/to/ros2_ws/src/coloact_robot/coloact_robot/Checkpoints/checkpoints_rgbd_elev/diffusion_final.pth' 
    # ===========================================

    return LaunchDescription([
        # 1. 鍚姩鐢垫満椹卞姩
        Node(
            package='coloact_robot',
            executable='motor_controller3_PWM_yibianman',
            name='motor_driver',
            output='screen'
        ),
        
        # 2. 鍚姩鎰熺煡 (Depth Anything)
        Node(
            package='coloact_robot',
            executable='depth_processing_node', 
            name='depth_processor',
            parameters=[
                {'camera_id': 0},
                {'processing_size': [320, 240]} 
            ]
        ),
        
        # 3. 鍚姩鎵嬫焺椹卞姩 (浠呯敤浜庡畨鍏ㄩ攣 L1)
        Node(
            package='joy_linux',
            executable='joy_linux_node',
            name='joy_node',
            parameters=[{'dev': '/dev/input/js0'}]
        ),

        # 4. 鍚姩鎵嬫焺閫昏緫 (鐢ㄤ簬鎶?L1 鏄犲皠鍒?/autonomous_enabled)
        # 娉ㄦ剰锛氳繖閲屾垜浠鐢ㄤ箣鍓嶇殑 joy_controller锛屽彧涓轰簡閭ｄ釜瀹夊叏閿佸姛鑳?
        Node(
            package='coloact_robot',
            executable='joy_controller3_for_inference_PWM', 
            name='joy_safety_switch',
            output='screen'
        ),
        
        # 5. 鍚姩 Diffusion 鎺ㄧ悊 (澶ц剳)
        Node(
            package='coloact_robot',
            executable='diffusion_inference_node',
            name='diffusion_pilot',
            output='screen',
            parameters=[
                {'input_mode': INPUT_MODE},
                {'auto_start': AUTO_START},
                {'model_path': MODEL_PATH}
            ]
        )
    ])

