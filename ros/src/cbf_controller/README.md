ECE 486 / 687 project code setup (Linux)
========================================

# Installation and testing

It is advisable to install and run all the code from within a [Docker](https://docs.docker.com/get-started/docker-overview/) container. This can be done following the instructions available [here](https://docs.docker.com/desktop/setup/install/linux/).

Full documentation of Colima can be found [here](https://oneuptime.com/blog/post/2026-02-08-how-to-install-docker-engine-without-docker-desktop-on-macos/view)


Once you have installed Docker, follow the steps below in order to run the `robomaster_ros` package within a Docker container.

**Note:** Until step 3 included, you need an internet connection, after which you have to switch to the following network, to which the robots automatically connect:
```
* SSID: brushbotarium
* password: brushbotarium
```

1. Install colima and docker with brew.

```
brew install colima docker docker-compose docker-buildx
```

2. Start colima. Make sure --network-mode=bridged. The parameters can be checked in the file ~/.colima/default/colima.yaml.
```
colima start --network-address --network-mode=bridged
```

If you get a warning:

WARN[0000] 'network mode' cannot be updated after initial setup, discarded 

Delete colima with the following command and redo step 2 

```
colima delete
```

3. In a terminal, navigate to a desired working directory (e.g. home, by running `cd`) and clone the `robomaster_ros` repository

```
git clone https://github.com/erablab/robomaster_ros.git
```

4. Navigate to the folder containing the Docker file for ROS2 Humble

```
cd robomaster_ros/docker/humble/
```

5. Build the Docker container

```
sudo docker build -f Dockerfile -t dji_robomaster_ros:1.0 .
```

6. Run the Docker container

```
sudo docker run -it --rm --network=host --pid=host --ipc=host --name="dji_robomaster_ros" dji_robomaster_ros:1.0
```

7. In the Docker container, source the ROS setup file and launch the ROS nodes required to control the robot

```
source /opt/ros/humble/setup.bash
source /opt/ros/ws/setup.bash
ros2 launch robomaster_ros ep.launch name:=robot8 serial_number:=3JKDH6C001J3H0


ros2 launch robomaster_ros ep.launch name:=robot9 serial_number:=3JKDH6C0012Z43

```

**Note:** The example above is to connect to the robot with ID 9. You will need to change the `serial_number` to match that of the robot you would like to control.

8. In another terminal, enter the running container

```
sudo docker exec -it dji_robomaster_ros /bin/bash
```

9. Source the ROS setup file, after which you can see topics, echo messages, etc.

```
source /opt/ros/humble/setup.bash
source /opt/ros/ws/setup.bash
ros2 topic list
ros2 topic echo /robot9/joint_states
ros2 topic pub /robot9/cmd_vel geometry_msgs/msg/Twist "{angular: {z: 1}}" --once
```

**Note:** You may also use a terminal multiplexer within the Docker container, such as [`screen`](https://www.gnu.org/software/screen/), instead of opening multiple terminals.

# Documentation

The full documentation of the ROS package is available online [here](https://jeguzzi.github.io/robomaster_ros/).




for jiaan only

docker compose -f docker-compose.dji_robomaster.yml up -d
docker exec -it dji_robomaster_ros bash

cd /opt/ros/my_ws/src/cbf_controller
colcon build --packages-select cbf_controller --symlink-install
source install/setup.bash


sudo docker rm -f dji_robomaster_ros   