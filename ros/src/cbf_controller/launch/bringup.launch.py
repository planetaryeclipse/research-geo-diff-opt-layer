from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(
        Node(
            package="cbf_controller",
            executable="robot_state_node",
            name="robot_state",
            output="screen",
        )
    )

    ld.add_action(
        Node(
            package="cbf_controller",
            executable="controller_node",
            name="controller",
            output="screen",
        )
    )

    return ld

