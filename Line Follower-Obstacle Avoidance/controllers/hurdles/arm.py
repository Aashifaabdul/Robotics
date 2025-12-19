# DO NOT MODIFY THIS FILE, IT'S CONSIDERED EXTERNAL TO THE ROBOT YOU ARE CONTROLLING
# DO NOT MODIFY THIS FILE, IT'S CONSIDERED EXTERNAL TO THE ROBOT YOU ARE CONTROLLING
# DO NOT MODIFY THIS FILE, IT'S CONSIDERED EXTERNAL TO THE ROBOT YOU ARE CONTROLLING

import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation as R

from controller import Supervisor

# DO NOT MODIFY THIS FILE, IT'S CONSIDERED EXTERNAL TO THE ROBOT YOU ARE CONTROLLING
# DO NOT MODIFY THIS FILE, IT'S CONSIDERED EXTERNAL TO THE ROBOT YOU ARE CONTROLLING
# DO NOT MODIFY THIS FILE, IT'S CONSIDERED EXTERNAL TO THE ROBOT YOU ARE CONTROLLING

class UR5e(Supervisor):
    def __init__(self):
        super().__init__()
        # set motor velocity, initialise sensors
        self.motors = [
            self.getDevice("shoulder_pan_joint"),
            self.getDevice("shoulder_lift_joint"),
            self.getDevice("elbow_joint"),
            self.getDevice("wrist_1_joint"),
            self.getDevice("wrist_2_joint"),
            self.getDevice("wrist_3_joint"),
        ]
        for finger in [self.getDevice('ROBOTIQ 2F-85 Gripper::left finger joint') ]:
            finger.setVelocity(0.8)
            finger.setTorque(finger.getAvailableTorque()/2)
        for m in self.motors:
            m.getPositionSensor().enable(12)
            m.setVelocity(0.1)
        # load the kinematic chain based on the robot's URDF file
        end_link = 'TCP'  # link used for forward and inverse kinematics
        # TCP = Tool Center Point - this is the point between the fingers
        URDF_FN = '../../resources/ur5e_2f85_camera.urdf'
        self.chain = kp.build_serial_chain_from_urdf(open(URDF_FN), end_link)
        
        # print chain on console
        print('kinematic chain:')
        print(self.chain)
        print(f'The end link of the chain is <{end_link}>.')
        print('All computations of forward and inverse kinematics apply to this link.')
        
        
    @property
    def home_pos(self):
        """ 
        this is the home configuration of the robot 
        """
        return [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]

    def joint_pos(self):
        """
        :return: (6,) ndarray, the current joint position of the robot
        """
        joint_pos = np.asarray([m.getPositionSensor().getValue() for m in self.motors])
        return joint_pos
        
    def move_to_joint_pos(self, target_joint_pos, timeout=5, velocity=0.8):
        """
        synchronised PTP motion
        blocking behaviour, moves the robot to the desired joint position.
        :param target_joint_pos: list/ndarray with joint configuration
        :param timeout: float, timeout in seconds after which this function returns latest
        :param velocity: float, target joint velocity in radians/second
        :return: bool, True if robot reaches the target position
                  else will return False after timeout (in seconds)
        """
        if len(target_joint_pos) != len(self.motors):
            raise ValueError('target joint configuration has unexpected length')
            
        abs_diffs = np.abs(target_joint_pos - self.joint_pos())
        velocity_gains = abs_diffs / np.max(abs_diffs)
            
        for pos, gain, motor in zip(target_joint_pos, velocity_gains, self.motors):
            motor.setPosition(pos)
            motor.setVelocity(gain * velocity)
            
        # step through simulation until timeout or position reache
        for step in range(int(timeout * 1000) // 12):
            self.step()

            # check if the robot is close enough to the target position
            if all(abs(target_joint_pos - self.joint_pos()) < 0.001):
                return True
                
        print('Timeout. Robot has not reached the desired target position.')
        return False
        
    def forward_kinematics(self, joint_pos=None):
        """
        computes the pose of the chain's end link for given joint position.
        :param joint_pos: joint position for which to compute the end-effector pose
                          if None given, will use the robot's current joint position
        :return: kinpy.Transform object with pos and rot
        """
        if joint_pos is None:
            joint_pos = self.joint_pos()
            
        ee_pose = self.chain.forward_kinematics(joint_pos)
        return ee_pose
        
    def inverse_kinematics(self, target_pose):
        """
        Computes a joint configuration to reach the given target pose.
        Note that the resulting joint position might not actually reach the target
        if the target is e.g. too far away.
        :param target_pose: kinpy.Transform, pose of the end link of the chain
        :return: list/ndarray, joint position
        """
        ik_result = self.chain.inverse_kinematics(target_pose, self.joint_pos())
        return ik_result