import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ================= 閰嶇疆鍖哄煙 =================
    # 榛樿妯″瀷璺緞 (璇风‘璁や綘鐨?.pth 鏂囦欢鍦ㄨ繖涓綅缃?
    default_model_path = '/path/to/ros2_ws/src/coloact_robot/coloact_robot/Checkpoints_ACT/act_final.pth'
    
    # 榛樿杈撳叆妯″紡 (蹇呴』涓庤缁冩椂涓€鑷?
    default_input_mode = 'rgbd_elev'
    
    # 鏄惁鑷姩鍚姩 (True=鐩存帴璺? False=闇€鎸夋墜鏌凩1閿?
    default_auto_start = 'False'
    # ===========================================

    # 澹版槑鍚姩鍙傛暟 (鍏佽鍛戒护琛岃鐩?
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value=default_model_path,
        description='Path to the ACT model checkpoint'
    )
    input_mode_arg = DeclareLaunchArgument(
        'input_mode', default_value=default_input_mode,
        description='Input mode: rgb, rgbd, or rgbd_elev'
    )
    auto_start_arg = DeclareLaunchArgument(
        'auto_start', default_value=default_auto_start,
        description='Enable autonomous mode on startup'
    )

    # 1. 搴曞眰鐢垫満椹卞姩鑺傜偣 (Motor Driver)
    motor_node = Node(
        package='coloact_robot',
        executable='motor_controller3_PWM_yibianman', # 浣犵殑鐢垫満椹卞姩鑺傜偣鍚?
        name='motor_driver',
        output='screen'
    )

    # 2. 娣卞害澶勭悊涓庤瑙夎妭鐐?(Depth Processing)
    # 璐熻矗浜х敓 /processed_depth 鍜?/processed_elevation
    depth_node = Node(
        package='coloact_robot',
        executable='depth_processing_node', 
        name='depth_processor',
        output='screen'
    )

    # 3. 鎵嬫焺鑺傜偣 (Joystick) - 鐢ㄤ簬瀹夊叏寮€鍏?(L1閿?
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'deadzone': 0.05}]
    )
    
    # 鎵嬫焺閫昏緫杞崲鑺傜偣 (鍙€夛紝濡傛灉浣犳湁涓撻棬鐨?joy_controller)
    # 杩欓噷鍋囪浣犵敤鎵嬫焺鎺у埗 start/stop锛屽鏋滀笉鍐茬獊鍙互淇濈暀
    joy_control_node = Node(
        package='coloact_robot',
        executable='joy_controller3_for_inference_PWM',
        name='joy_safety_switch',
        output='screen'
    )

    # 4. 馃敟 ACT 鎺ㄧ悊鑺傜偣 (The Core) 馃敟
    act_node = Node(
        package='coloact_robot',
        executable='act_inference_node', # 蹇呴』鍦?setup.py 閲岄厤缃繃
        name='act_pilot',
        output='screen',
        parameters=[
            {'model_path': LaunchConfiguration('model_path')},
            {'input_mode': LaunchConfiguration('input_mode')},
            {'auto_start': LaunchConfiguration('auto_start')}
        ]
    )

    return LaunchDescription([
        model_path_arg,
        input_mode_arg,
        auto_start_arg,
        motor_node,
        depth_node,
        joy_node,
        joy_control_node,
        act_node
    ])

