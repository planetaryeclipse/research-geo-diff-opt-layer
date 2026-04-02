import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class RobotStateNode(Node):
    """
    Integrates commanded accelerations into commanded velocities.

    Subscribes:
      - `commanded_accels` : Float64MultiArray.data = [accel, angular_accel]

    Publishes:
      - `/robot9/cmd_vel` : geometry_msgs/Twist

    Integration rule (trapezoidal):
      v_n = v_{n-1} + (a_{n-1} + a_n)/2 * dt
    """

    def __init__(self) -> None:
        super().__init__("robot_state")

        self.declare_parameter("accels_topic", "commanded_accels")
        self.declare_parameter("cmd_vel_topic", "/robot8/cmd_vel")



        accels_topic = str(self.get_parameter("accels_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self._linear_vel = 0.0
        self._angular_vel = 0.0

        self._max_dt = 1.0

        self._last_time_s: Optional[float] = None
        self._prev_accel: Optional[float] = None
        self._prev_ang_accel: Optional[float] = None

        self._sub = self.create_subscription(
            Float64MultiArray, accels_topic, self._on_commanded_accels, 10
        )
        self._pub = self.create_publisher(Twist, cmd_vel_topic, 10)

    def _on_commanded_accels(self, msg: Float64MultiArray) -> None:
        if msg.data is None or len(msg.data) < 2:
            self.get_logger().warn(
                "`commanded_accels` must contain at least 2 elements: "
                "[acceleration, angular_acceleration]"
            )
            return

        commanded_accel = float(msg.data[0])
        commanded_ang_accel = float(msg.data[1])

        now_s = time.monotonic()

        # First message: only store the acceleration so next callback can compute dt.
        if self._last_time_s is None or self._prev_accel is None or self._prev_ang_accel is None:
            self._last_time_s = now_s
            self._prev_accel = commanded_accel
            self._prev_ang_accel = commanded_ang_accel

            twist = Twist()
            twist.linear.x = float(self._linear_vel)
            twist.angular.z = float(self._angular_vel)
            self._pub.publish(twist)
            return

        dt = now_s - self._last_time_s
        if self._last_time_s > self._max_dt:
            print("Max dt surpassed. Clamping to t=", self._max_dt)
            dt = self._max_dt


        # v_n = v_{n-1} + (a_{n-1} + a_n)/2 * dt

        self._linear_vel = self._linear_vel + (self._prev_accel + commanded_accel) * 0.5 * dt
        self._angular_vel = (
                self._angular_vel + (self._prev_ang_accel + commanded_ang_accel) * 0.5 * dt
        )

        # Shift state forward for next integration step.
        self._last_time_s = now_s
        self._prev_accel = commanded_accel
        self._prev_ang_accel = commanded_ang_accel

        twist = Twist()
        twist.linear.x = float(self._linear_vel)
        twist.angular.z = float(self._angular_vel)
        self._pub.publish(twist)


def main() -> None:
    rclpy.init()
    node = RobotStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
