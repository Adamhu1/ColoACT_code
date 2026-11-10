from launch import LaunchDescription
from launch_ros.actions import Node
import os
import subprocess
from launch.actions import ExecuteProcess
from launch.substitutions import EnvironmentVariable

def generate_launch_description():
    # 璁剧疆鐜鍙橀噺锛岀‘淇濅娇鐢ㄦ纭殑conda鐜
    conda_env_path = "/path/to/conda/env"
    python_path = f"{conda_env_path}/bin/python3.8"

    # 璁剧疆鐜鍙橀噺
    env_vars = {
        'PYTHONPATH': "/path/to/ros2_ws/install/coloact_robot/lib/python3.8/site-packages",
        'LD_LIBRARY_PATH': f"{conda_env_path}/lib",
    }

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
            }],
            output='screen'
        ),

        # 2. 鍚姩娣卞害澶勭悊鑺傜偣 (浣跨敤conda鐜)
        ExecuteProcess(
            cmd=[python_path, '/path/to/ros2_ws/src/coloact_robot/coloact_robot/w_danmu_xiangdui_gaochengtu.py',
                 '--ros-args', '-r', '__node:=depth_processor',
                 '-p', 'camera_id:=1', '-p', 'debug_mode:=false'],
            name='depth_processor',
            output='screen',
            env=env_vars
        ),

        # 3. 鍚姩鎵嬫焺鎺у埗鍣?(浣跨敤conda鐜)
        ExecuteProcess(
            cmd=[python_path, '/path/to/ros2_ws/src/coloact_robot/coloact_robot/joy_controller3_PWM.py',
                 '--ros-args', '-r', '__node:=joy_controller'],
            name='joy_controller',
            output='screen',
            env=env_vars
        ),

        # 4. 鍚姩鐢垫満椹卞姩 (浣跨敤conda鐜)
        ExecuteProcess(
            cmd=[python_path, '/path/to/ros2_ws/src/coloact_robot/coloact_robot/motor_controller3_PWM_yibianman.py',
                 '--ros-args', '-r', '__node:=motor_driver'],
            name='motor_driver',
            output='screen',
            env=env_vars
        ),

        # 5. 鍚姩鏁版嵁閲囬泦鑺傜偣 (浣跨敤鐩存帴瑙ｆ瀽鐗堟湰锛岄伩鍏峜v_bridge搴撳啿绐?
        ExecuteProcess(
            cmd=[python_path, '/path/to/ros2_ws/src/coloact_robot/coloact_robot/data_collector_direct.py',
                 '--ros-args', '-r', '__node:=data_collector'],
            name='data_collector',
            output='screen',
            env=env_vars
        )
    ])

