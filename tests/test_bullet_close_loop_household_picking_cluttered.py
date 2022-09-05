import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletHouseholdPickingCluttered(unittest.TestCase):
  env_config = {}

  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 15, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (0.8, 0.8),
                'view_type': 'camera_center_xyz', 'hard_reset_freq': 1000, 'z_termination': True}
  planner_config = {'random_orientation': False, 'dpos': 0.05, 'drot': np.pi/8}
  env_config['seed'] = 1
  # env = CloseLoopHouseholdPickingClutteredEnv(env_config)
  # planner = CloseLoopHouseholdPickingClutteredPlanner(env, planner_config)

  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['seed'] = 0
    num_processes = 10
    env = env_factory.createEnvs(num_processes,  'close_loop_clutter_picking', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=500)
    while total < 500:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)
      # plt.imshow(obs_[0, 0], vmin=0, vmax=0.2)
      # plt.show()
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{:.3f}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
    env.close()

