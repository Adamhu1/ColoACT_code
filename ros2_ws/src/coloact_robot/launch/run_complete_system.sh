#!/bin/bash

# 完整数据采集系统启动脚本
# 这个脚本会使用正确的conda环境启动所有节点

echo "🚀 启动完整的数据采集系统..."
echo "数据将保存在: ~/ros2_ws/src/coloact_robot/dataset/"
echo ""
echo "操作说明:"
echo "  - Xbox手柄 [START] 键开始录制 (按键索引 7)"
echo "  - Xbox手柄 [BACK/SELECT] 键停止录制 (按键索引 6)"
echo "  - 按 Ctrl+C 停止整个系统"
echo ""

# 确保我们有正确的conda环境
CONDA_ENV="/path/to/conda/env"
ROS2_WS="/path/to/ros2_ws"

# 设置环境变量
export PYTHONPATH="${ROS2_WS}/install/coloact_robot/lib/python3.8/site-packages:$PYTHONPATH"
export ROS_DOMAIN_ID=0

# 启动节点的函数
start_node() {
    local node_name=$1
    local script_path=$2
    local ros_args=$3

    echo "启动节点: $node_name"
    ${CONDA_ENV}/bin/python3 $script_path $ros_args &
}

# 创建日志目录
mkdir -p ~/.ros/log

# 1. 启动手柄驱动 (joy_linux)
echo "🎮 启动手柄驱动..."
ros2 run joy_linux joy_linux_node --ros-args -p dev:=/dev/input/js0 -p deadzone:=0.05 -p autorepeat_rate:=20.0 &
JOY_PID=$!

# 等待一下让手柄驱动启动
sleep 2

# 2. 启动深度处理节点
echo "📷 启动深度处理节点..."
start_node "depth_processor" "${ROS2_WS}/src/coloact_robot/coloact_robot/w_danmu_xiangdui_gaochengtu.py" "--ros-args -r __node:=depth_processor -p camera_id:=0 -p debug_mode:=false"
DEPTH_PID=$!

# 等待深度节点启动
sleep 5

# 3. 启动手柄控制器
echo "🕹️ 启动手柄控制器..."
start_node "joy_controller" "${ROS2_WS}/src/coloact_robot/coloact_robot/joy_controller3_PWM.py" "--ros-args -r __node:=joy_controller"
JOY_CTRL_PID=$!

# 4. 启动电机控制器
echo "⚙️ 启动电机控制器..."
start_node "motor_driver" "${ROS2_WS}/src/coloact_robot/coloact_robot/motor_controller3_PWM_yibianman.py" "--ros-args -r __node:=motor_driver"
MOTOR_PID=$!

# 等待系统稳定
sleep 3

# 5. 启动数据采集节点 (使用直接解析版本，避免cv_bridge库冲突)
echo "💾 启动数据采集节点..."
start_node "data_collector" "${ROS2_WS}/src/coloact_robot/coloact_robot/data_collector_direct.py" "--ros-args -r __node:=data_collector"
DATA_PID=$!

echo ""
echo "✅ 系统启动完成!"
echo ""
echo "🎯 系统状态检查:"
echo "   - 手柄驱动: 运行中 (PID: $JOY_PID)"
echo "   - 深度处理: 运行中 (PID: $DEPTH_PID)"
echo "   - 手柄控制: 运行中 (PID: $JOY_CTRL_PID)"
echo "   - 电机控制: 运行中 (PID: $MOTOR_PID)"
echo "   - 数据采集: 运行中 (PID: $DATA_PID)"
echo ""
echo "📝 现在可以开始录制数据了!"
echo "   按手柄的 [START] 键开始录制，[BACK] 键停止录制"
echo ""

# 等待用户中断
cleanup() {
    echo ""
    echo "🛑 正在关闭系统..."

    # 杀死所有子进程
    if [ ! -z "$JOY_PID" ]; then kill $JOY_PID 2>/dev/null; fi
    if [ ! -z "$DEPTH_PID" ]; then kill $DEPTH_PID 2>/dev/null; fi
    if [ ! -z "$JOY_CTRL_PID" ]; then kill $JOY_CTRL_PID 2>/dev/null; fi
    if [ ! -z "$MOTOR_PID" ]; then kill $MOTOR_PID 2>/dev/null; fi
    if [ ! -z "$DATA_PID" ]; then kill $DATA_PID 2>/dev/null; fi

    # 额外确保进程被杀死
    pkill -f "joy_linux_node" 2>/dev/null
    pkill -f "w_danmu_xiangdui_gaochengtu.py" 2>/dev/null
    pkill -f "joy_controller3_PWM.py" 2>/dev/null
    pkill -f "motor_controller3_PWM_yibianman.py" 2>/dev/null
    pkill -f "data_collector.py" 2>/dev/null

    echo "✅ 系统已关闭"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 保持脚本运行
while true; do
    sleep 1
done