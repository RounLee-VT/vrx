#!/usr/bin/env python3
import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

BANNER = """
==================================================
           VRX WAM-V Keyboard Teleop
==================================================
  [W] : Forward (+)           [S] : Backward (-)
  [A] : Turn Left             [D] : Turn Right
  [Q] : Left Thruster CCW     [E] : Right Thruster CW
  [SPACE] : Emergency Stop (Thrust 0)
  [CTRL+C] : Exit
==================================================
"""

class VRXTeleopNode(Node):
    def __init__(self):
        super().__init__('vrx_teleop_node')
        
        # Publisher setting
        self.pub_left_thrust = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_right_thrust = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.pub_left_pos = self.create_publisher(Float64, '/wamv/thrusters/left/pos', 10)
        self.pub_right_pos = self.create_publisher(Float64, '/wamv/thrusters/right/pos', 10)

        # Variable initialization
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        self.angle = 0.0

        # Control parameters
        self.thrust_step = 10.0   # N unit increment/decrement
        self.max_thrust = 250.0   # Maximum thrust (N)
        self.angle_step = 0.05    # Radian unit increment/decrement (~2.8 degrees)
        self.max_angle = 0.785    # Maximum angle (~45 degrees)

    def publish_commands(self):
        self.pub_left_thrust.publish(Float64(data=float(self.left_thrust)))
        self.pub_right_thrust.publish(Float64(data=float(self.right_thrust)))
        self.pub_left_pos.publish(Float64(data=float(self.angle)))
        self.pub_right_pos.publish(Float64(data=float(self.angle)))

    def stop(self):
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        self.angle = 0.0
        self.publish_commands()

    def update_control(self, key):
        if key == 'w':
            self.left_thrust = min(self.max_thrust, self.left_thrust + self.thrust_step)
            self.right_thrust = min(self.max_thrust, self.right_thrust + self.thrust_step)
        elif key == 's':
            self.left_thrust = max(-self.max_thrust, self.left_thrust - self.thrust_step)
            self.right_thrust = max(-self.max_thrust, self.right_thrust - self.thrust_step)
        elif key == 'a':
            self.left_thrust = max(-self.max_thrust, self.left_thrust - self.thrust_step)
            self.right_thrust = min(self.max_thrust, self.right_thrust + self.thrust_step)
        elif key == 'd':
            self.left_thrust = min(self.max_thrust, self.left_thrust + self.thrust_step)
            self.right_thrust = max(-self.max_thrust, self.right_thrust - self.thrust_step)
        elif key == 'q':
            self.angle = min(self.max_angle, self.angle + self.angle_step)
        elif key == 'e':
            self.angle = max(-self.max_angle, self.angle - self.angle_step)
        elif key == ' ':
            self.stop()
        
        self.publish_commands()
        self.print_status()

    def print_status(self):
        print(f"\r[Output] Left: {self.left_thrust:6.1f} N | Right: {self.right_thrust:6.1f} N | Angle: {self.angle:5.2f} rad   ", end='', flush=True)

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = VRXTeleopNode()

    print(BANNER)
    node.print_status()

    try:
        while rclpy.ok():
            key = get_key(settings)
            if key == '\x03':  # Ctrl+C
                break
            elif key != '':
                node.update_control(key)
            rclpy.spin_once(node, timeout_sec=0.01)
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        node.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()
        print("\nController has been terminated.")

if __name__ == '__main__':
    main()