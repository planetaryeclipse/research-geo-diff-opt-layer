from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ControllerPlaceholder(Node):
    """
    Placeholder controller node.

    It subscribes to `robot_state` and publishes a dummy `control_cmd`.
    Replace this logic later with your real controller / optimization pipeline.
    """

    def __init__(self):
        super().__init__("controller_placeholder")

        self.declare_parameter("update_rate_hz", 10.0)
        rate_hz = float(self.get_parameter("update_rate_hz").value)

        self._last_state: Optional[str] = None

        self._sub = self.create_subscription(
            String, "robot_state", self._on_state, 10
        )
        self._pub = self.create_publisher(String, "control_cmd", 10)

        period_s = 1.0 / max(rate_hz, 1e-6)
        self._timer = self.create_timer(period_s, self._on_timer)

    def _on_state(self, msg: String):
        self._last_state = msg.data

    def _on_timer(self):
        cmd = String()
        cmd.data = (
            f"{{'controller': 'placeholder', 'using_robot_state': "
            f"{self._last_state is not None}}}"
        )
        self._pub.publish(cmd)
        self.get_logger().debug(
            f"Published control_cmd (robot_state_received={self._last_state is not None})"
        )


def main():
    rclpy.init()
    node = ControllerPlaceholder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
