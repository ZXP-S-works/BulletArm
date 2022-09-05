import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class OpenCloseLoopRLHouseholdPickingClutteredPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.current_target = None
    self.target_object = None
    self.stage = None
    self.view_type = config['view_type'] if 'view_type' in config.keys() else None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if self.stage == 'pick' and \
            (np.all(np.abs([x, y, z]) < self.dpos / 2) and np.abs(r) < self.drot / 2 or self.loop > 20):
      primitive = constants.PICK_PRIMATIVE
      self.stage = 'lift'
      self.current_target[0][2] = 0.25
    else:
      primitive = constants.PICK_PRIMATIVE if self.stage == 'lift' else constants.PLACE_PRIMATIVE
    self.loop += 1
    return self.env._encodeAction(primitive, x, y, z, r)

  def getNextAction(self, posture=None):
    if self.env.current_episode_steps == 0\
            or self.env.grasp_done == 1:
      if posture is None:
        objects = np.array(list(filter(lambda x: not self.isObjectHeld(x) and self.isObjOnTop(x), self.env.objects)))
        np.random.shuffle(objects)
        self.target_object = objects[0]
        object_pos = list(self.target_object.getPosition())
        object_pos[2] += (np.random.random() - 1) * 0.02
        object_rot = list(transformations.euler_from_quaternion(self.target_object.getRotation()))
        rz = (np.random.random() - 0.5) * np.pi
        object_rot[2] = rz
        posture = (object_pos, object_rot)

      self.current_target = posture
      self.stage = 'pick'
      self.loop = 0
    return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100