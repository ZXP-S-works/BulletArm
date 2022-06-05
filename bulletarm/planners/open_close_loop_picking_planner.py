import numpy as np

from bulletarm.planners.close_loop_household_picking_cluttered_planner import CloseLoopHouseholdPickingClutteredPlanner
from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class OpenCloseLoopPickingPlanner(CloseLoopHouseholdPickingClutteredPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def setNewTarget(self, obj=None):
    if self.stage == 0:
      if obj is not None:
        self.target_object = obj
      else:
        object_pos = self.env.robot._getEndEffectorPosition()
        object_pos[2] -= 0.05
        object_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
        self.current_target = (object_pos, object_rot, constants.PICK_PRIMATIVE)
        self.stage = 1
    else:
      object_pos = self.env.robot._getEndEffectorPosition()
      object_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
      self.current_target = ((object_pos[0], object_pos[1], 0.21), object_rot, constants.PICK_PRIMATIVE)
      self.stage = 0
