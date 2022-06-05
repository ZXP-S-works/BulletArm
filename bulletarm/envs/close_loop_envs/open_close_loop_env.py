import pybullet as pb
import numpy as np
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils.renderer import Renderer
from bulletarm.pybullet.utils.ortho_sensor import OrthographicSensor
from bulletarm.pybullet.utils.sensor import Sensor
from bulletarm.pybullet.equipments.tray import Tray
from scipy.ndimage import rotate
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
import bulletarm.pybullet.utils.constants as constants


class OpenCloseLoopEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    if config['robot'] == 'kuka':  # default in kuka.py
      self.robot.home_positions = [0.3926, 0., -2.137, 1.432, 0, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0.]
      self.robot.home_positions_joint = self.robot.home_positions[:7]


  def step(self, action):
    p, x, y, z, rot = self._decodeAction(action)
    if p == constants.TELEPORT_PRIMSTIVE:  # open loop control
      self.teleport(action)
      reward = 0
      done = False
      pos = np.array([x, y, z])
      rot = np.array(rot)
    else:  # close loop control
      current_pos = self.robot._getEndEffectorPosition()
      current_rot = list(transformations.euler_from_quaternion(self.robot._getEndEffectorRotation()))
      if self.action_sequence.count('r') == 1:
        current_rot[0] = 0
        current_rot[1] = 0

      # bTg = transformations.euler_matrix(0, 0, current_rot[-1])
      # bTg[:3, 3] = current_pos
      # gTt = np.eye(4)
      # gTt[:3, 3] = [x, y, z]
      # bTt = bTg.dot(gTt)
      # pos = bTt[:3, 3]

      pos = np.array(current_pos) + np.array([x, y, z])
      rot = np.array(current_rot) + np.array(rot)
      rot_q = pb.getQuaternionFromEuler(rot)
      pos[0] = np.clip(pos[0], self.workspace[0, 0], self.workspace[0, 1])
      pos[1] = np.clip(pos[1], self.workspace[1, 0], self.workspace[1, 1])
      pos[2] = np.clip(pos[2], self.workspace[2, 0], self.workspace[2, 1])
      self.robot.moveTo(pos, rot_q, dynamic=True)
      self.robot.controlGripper(p)
      self.robot.adjustGripperCommand()
      self.setRobotHoldingObj()

    self.renderer.clearPoints()
    obs = self._getObservation(action)
    valid = self.isSimValid()
    if valid:
      done = self._checkTermination()
      # reward = 1.0 if done else 0.0
      reward = 0
    else:
      done = True
      reward = 0
    if not done:
      done = self.current_episode_steps >= self.max_steps
    self.current_episode_steps += 1

    self.simulate_pos = pos
    self.simulate_rot = rot
    return obs, reward, done


  def resetPybulletWorkspace(self):
    self.renderer.clearPoints()
    BaseEnv.resetPybulletWorkspace(self)
    self.simulate_pos = self.robot._getEndEffectorPosition()
    self.simulate_rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())


  def _getTopDownObs(self):
    self.initSensor()
    heightmap =  self.sensor.getHeightmap(self.heightmap_size)
    heightmap = heightmap.reshape([1, self.heightmap_size, self.heightmap_size])
    return self._isHolding(), None, heightmap