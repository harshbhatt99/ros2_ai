# ROS2 + AI Packages

This repository contains different packages that uses AI on ROS2. The AI packages can be used for different applications in Robotics and Industrial Automation.

### Dependencies

* ROS2 Humble
* Ubuntu 22.04

### Installing

* Install PyTorch
```
pip install torch
```
* Clone the repository under src folder of your ROS workspace.
* Change current directory to ROS workspace and run "colcon build".
* Source the setup.bash file in current ROS2 workspace.
```
source install/setup.bash
```

### Executing program

* Run autoencoder_node
```
ros2 run ros2_autoencoder autoencoder_node
```
* Open another terminal and send the messsage on number_callback topic
```
ros2 topic pub /number_input std_msgs/msg/Float64 "{data: 1.5}"
```
