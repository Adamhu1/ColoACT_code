from glob import glob
import os

from setuptools import find_packages, setup


package_name = "coloact_robot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "h5py", "numpy", "opencv-python", "pyserial"],
    zip_safe=True,
    description="ColoACT ROS 2 deployment and robot-control nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "act_inference_node = coloact_robot.ACT_inference_node:main",
            "bc_singlestep_inference_node = coloact_robot.bc_singlestep_inference_node:main",
            "data_collector = coloact_robot.run_data_collector:main",
            "data_collector_direct = coloact_robot.data_collector_direct:main",
            "data_collector_labeled = coloact_robot.data_collector_labeled:main",
            "depth_processing_node = coloact_robot.w_danmu_xiangdui_gaochengtu:main",
            "diffusion_inference_node = coloact_robot.diffusion_inference_node:main",
            "diffusion_two_channel_inference = coloact_robot.diffusion_2tongdao_inference:main",
            "joy_controller = coloact_robot.joy_controller:main",
            "joy_controller3_PWM = coloact_robot.joy_controller3_PWM:main",
            "joy_controller3_for_inference_PWM = coloact_robot.joy_controller3_for_inference_PWM:main",
            "joy_controller4_PWM_double_joy = coloact_robot.joy_controller4_PWM_double_joy:main",
            "joy_controller_labeled = coloact_robot.joy_controller_labeled:main",
            "motor_controller = coloact_robot.motor_controller:main",
            "motor_controller2_PWM = coloact_robot.motor_controller2_PWM:main",
            "motor_controller3_PWM = coloact_robot.motor_controller3_PWM:main",
            "motor_controller3_PWM_yibianman = coloact_robot.motor_controller3_PWM_yibianman:main",
        ],
    },
)
