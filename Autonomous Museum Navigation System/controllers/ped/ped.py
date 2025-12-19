# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pedestrian class container."""
import math
import random
import numpy as np

from controller import Robot
from controller import Supervisor


class Pedestrian(Supervisor):
    """Control a Pedestrian PROTO."""

    def __init__(self):
        """Constructor: initialize constants."""
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.speed = 0.0008
        self.current_height_offset = 0
        self.height_offsets = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03 ]
        Supervisor.__init__(self)
        self.scared = self.getTime() - 10.


    def run(self):
        self.time_step = int(self.getBasicTimeStep())
        self.root_node_ref = self.getSelf()
        
        while not self.step(self.time_step) == -1:
            ttime = self.getTime()

            # Get robot position
            xx_node = self.getFromDef("MECA")
            robot_t = np.array(xx_node.getField("translation").getSFVec3f())

            # Get pedestrian position
            self.root_translation_field = self.root_node_ref.getField("translation")
            self.root_rotation_field = self.root_node_ref.getField("rotation")
            ped_t = np.array(self.root_node_ref.getField("translation").getSFVec3f())

            # Pioneers
            for i in range(2):
                pioneer_node = self.getFromDef("BYE"+ str(i+1))
                dst = np.array(pioneer_node.getField("translation").getSFVec3f()[:-1]) - ped_t[:-1]
                if np.linalg.norm(dst) < 2.0:
                    self.scared = self.getTime()
                    rx = random.uniform(-3, 4)
                    while abs(rx) < 0.5:
                        rx = random.uniform(-4, 4)
                    ry = random.uniform(-5, 5)
                    while abs(ry) < 0.5:
                        ry = random.uniform(-5, 5)
                    ra = random.uniform(0, 2*math.pi)
                    self.root_translation_field.setSFVec3f([rx, ry, self.ROOT_HEIGHT + self.current_height_offset])
                    self.root_rotation_field.setSFRotation([0, 0, 1, ra])
                    break

            # print(time.time() - self.scared)
            if self.getTime() - self.scared > 5.0:
                seq = int(((ttime * self.speed) / self.CYCLE_TO_DISTANCE_RATIO) % len(self.height_offsets))
                ratio = (ttime * self.speed) / self.CYCLE_TO_DISTANCE_RATIO - \
                    int(((ttime * self.speed) / self.CYCLE_TO_DISTANCE_RATIO))
                self.current_height_offset = self.height_offsets[seq] * (1 - ratio) + \
                    self.height_offsets[(seq + 1) % len(self.height_offsets)] * ratio

                error = robot_t - ped_t
                error[2] = 0  # ignore z axis
                distance = np.linalg.norm(error)
                direction = error/distance
                inc = direction * self.speed * self.time_step
                if distance > 1.2:
                    move = True
                    ped_t2 = ped_t + inc
                elif distance < 1.0:
                    move = True
                    ped_t2 = ped_t - inc
                else:
                    move = False
                    ped_t2 = ped_t
                ped_t2o = ped_t + inc


                self.waypoints = [ped_t, ped_t2]
                x = distance * self.waypoints[1][0] + (1 - distance) * self.waypoints[0][0]
                y = distance * self.waypoints[1][1] + (1 - distance) * self.waypoints[0][1]
                root_translation = [x, y, self.ROOT_HEIGHT + self.current_height_offset]
                if move:
                    self.root_translation_field.setSFVec3f(root_translation)
                
                
                self.waypoints = [ped_t, ped_t2o]
                angle = math.atan2(self.waypoints[1][1] - self.waypoints[0][1],
                                self.waypoints[1][0] - self.waypoints[0][0])
                if angle != 0:
                    self.root_rotation_field.setSFRotation([0, 0, 1, angle])

    

controller = Pedestrian()
controller.run()
