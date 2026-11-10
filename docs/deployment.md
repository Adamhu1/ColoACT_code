# ROS 2 deployment

The ROS 2 package is under `ros2_ws/src/coloact_robot`. It contains perception,
ACT inference, data collection, joystick, and motor-control nodes.

Before deployment, verify camera topics, model paths, serial-port settings, and
the motor emergency-stop behavior. Hardware-specific values must remain local
and must not be committed.

