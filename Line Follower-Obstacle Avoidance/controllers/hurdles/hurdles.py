import math
import numpy as np

from scipy.spatial.transform import Rotation as R

import kinpy as kp

from arm import UR5e


def main():
    robot = UR5e()
    robot.move_to_joint_pos(robot.home_pos)
    steps = 0
    while robot.step(25) != -1:
        tf_target = kp.Transform(
            pos = [0, 0.5+0.25*math.sin(steps), 0.03],
            rot = [-np.pi/2, 0, 0]
        )
        robot.move_to_joint_pos(robot.inverse_kinematics(tf_target))
        steps += 0.05

if __name__ == '__main__':
    main()
