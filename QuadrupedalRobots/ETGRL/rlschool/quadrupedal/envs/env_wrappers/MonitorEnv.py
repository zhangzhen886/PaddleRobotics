import os
import gym
import numpy as np
import sys
import pybullet
from rlschool.quadrupedal.robots import robot_config
from rlschool import quadrupedal
from rlschool.quadrupedal.robots import action_filter
from rlschool.quadrupedal.envs.utilities.ETG_model import ETG_layer, ETG_model
from copy import copy

Param_Dict = {
  'rew_torso'       : 0.0,
  'rew_up'          : 0.6,
  'rew_feet_vel'    : 0.0,
  'rew_tau'         : 0.2,
  'rew_done'        : 1,
  'rew_velx'        : 0,
  'rew_actionrate'  : 0.2,
  'rew_badfoot'     : 0.1,
  'rew_footcontact' : 0.1,
  'rew_feet_airtime': 0.0,
  'rew_feet_pos'    : 2.0
}
Random_Param_Dict = {
  'random_dynamics': 0,
  'random_force'   : 0
}


# call in A1GymEnv.init(), behind "build_regular_env()"
def EnvWrapper(env, reward_param, sensor_mode, normal=0, ETG_T=0.5, enable_action_filter=False,
               reward_p=1, ETG=1, ETG_path="", ETG_T2=0.5, random_param=None,
               ETG_H=20, act_mode="traj", vel_d=0.6, vel_mode="max", foot_dx=0.2,
               task_mode="normal", step_y=0.05):
  env = ETGWrapper(env=env, ETG=ETG, ETG_T=ETG_T, ETG_path=ETG_path,
                   ETG_T2=ETG_T2, ETG_H=ETG_H, act_mode=act_mode,
                   task_mode=task_mode, step_y=step_y)
  env = ActionFilterWrapper(env=env, enable_action_filter=enable_action_filter)
  env = RandomWrapper(env=env, random_param=random_param)
  env = ObservationWrapper(env=env, ETG=ETG, sensor_mode=sensor_mode, normal=normal, ETG_H=ETG_H)
  env = RewardShaping(env=env, reward_param=reward_param, reward_p=reward_p, vel_d=vel_d, vel_mode=vel_mode, foot_dx=foot_dx)
  return env


class ActionFilterWrapper(gym.Wrapper):
  def __init__(self, env, enable_action_filter):
    gym.Wrapper.__init__(self, env)
    self.robot = self.env.robot
    self.pybullet_client = self.env.pybullet_client
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.enable_action_filter = enable_action_filter and self.env.ETG.endswith("sac")
    if self.enable_action_filter:
      self._action_filter = self._BuildActionFilter()

  def reset(self, **kwargs):
    obs_all, info = self.env.reset(**kwargs)
    self._step_counter = 0
    if self.enable_action_filter:
      self._ResetActionFilter()
    return obs_all, info

  def step(self, action, **kwargs):
    if self.enable_action_filter:
      action = self._FilterAction(action)
    obs_all, rew, done, info = self.env.step(action)
    self._step_counter += 1
    return obs_all, rew, done, info

  def _BuildActionFilter(self):
    sampling_rate = 1 / self.env.env_time_step
    num_joints = 12
    a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,
                                                num_joints=num_joints)
    return a_filter

  def _ResetActionFilter(self):
    self._action_filter.reset()

  def _FilterAction(self, action):
    # initialize the filter history, since resetting the filter will fill
    # the history with zeros and this can cause sudden movements at the start
    # of each episode
    if self._step_counter == 0:
      default_action = np.array([0, 0, 0] * 4)
      self._action_filter.init_history(default_action)
      # for j in range(10):
      #     self._action_filter.filter(default_action)

    filtered_action = self._action_filter.filter(action)
    # print(filtered_action)
    return filtered_action


class ObservationWrapper(gym.Wrapper):
  def __init__(self, env, ETG, sensor_mode, normal, ETG_H):
    gym.Wrapper.__init__(self, env)
    # print("env_time:",self.env.env_time_step)
    self.robot = self.env.robot
    self.pybullet_client = self.env.pybullet_client
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.sensor_mode = sensor_mode
    self.normal = normal
    self.ETG_H = ETG_H
    self.ETG = ETG
    self.ETG_mean = np.array([2.1505982e-02, 3.6674485e-02, -6.0444288e-02,
                              2.4625482e-02, 1.5869144e-02, -3.2513142e-02, 2.1506395e-02,
                              3.1869926e-02, -6.0140789e-02, 2.4625063e-02, 1.1628972e-02,
                              -3.2163858e-02])
    self.ETG_std = np.array([4.5967497e-02, 2.0340437e-01, 3.7410179e-01, 4.6187632e-02, 1.9441207e-01, 3.9488649e-01,
                             4.5966785e-02, 2.0323379e-01, 3.7382501e-01, 4.6188373e-02, 1.9457331e-01, 3.9302582e-01])
    if self.ETG:
      if "ETG" in self.sensor_mode.keys() and sensor_mode["ETG"]:
        sensor_shape = self.observation_space.high.shape[0]
        obs_h = np.array([1] * (sensor_shape + 12))
        obs_l = np.array([0] * (sensor_shape + 12))
        self.observation_space = gym.spaces.Box(obs_l, obs_h, dtype=np.float32)

      if "ETG_obs" in self.sensor_mode.keys() and sensor_mode["ETG_obs"]:
        sensor_shape = self.observation_space.high.shape[0]
        obs_h = np.array([1] * (sensor_shape + self.ETG_H))
        obs_l = np.array([0] * (sensor_shape + self.ETG_H))
        self.observation_space = gym.spaces.Box(obs_l, obs_h, dtype=np.float32)

    if "force_vec" in self.sensor_mode.keys() and sensor_mode["force_vec"]:
      sensor_shape = self.observation_space.high.shape[0]
      obs_h = np.array([1] * (sensor_shape + 6))
      obs_l = np.array([0] * (sensor_shape + 6))
      self.observation_space = gym.spaces.Box(obs_l, obs_h, dtype=np.float32)

    if "dynamic_vec" in self.sensor_mode.keys() and sensor_mode["dynamic_vec"]:
      sensor_shape = self.observation_space.high.shape[0]
      obs_h = np.array([1] * (sensor_shape + 3))
      obs_l = np.array([0] * (sensor_shape + 3))
      self.observation_space = gym.spaces.Box(obs_l, obs_h, dtype=np.float32)

    if "yaw" in self.sensor_mode.keys() and sensor_mode["yaw"]:
      sensor_shape = self.observation_space.high.shape[0]
      obs_h = np.array([1] * (sensor_shape + 2))
      obs_l = np.array([0] * (sensor_shape + 2))
      self.observation_space = gym.spaces.Box(obs_l, obs_h, dtype=np.float32)

    if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"] > 0:
      self.time_steps = sensor_mode["RNN"]["time_steps"]
      self.time_interval = sensor_mode["RNN"]["time_interval"]
      self.sensor_shape = self.observation_space.high.shape[0]
      self.obs_history = np.zeros((self.time_steps * self.time_interval, self.sensor_shape))
      if sensor_mode["RNN"]["mode"] == "stack":
        obs_h = np.array([1] * (self.sensor_shape * (self.time_steps + 1)))
        obs_l = np.array([0] * (self.sensor_shape * (self.time_steps + 1)))
        self.observation_space = gym.spaces.Box(obs_l, obs_h, dtype=np.float32)

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    self.dynamic_info = info["dynamics"]
    if self.ETG:
      if "ETG" in self.sensor_mode.keys() and self.sensor_mode["ETG"]:
        ETG_out = info["ETG_act"]
        if self.normal:
          ETG_out = (ETG_out - self.ETG_mean) / self.ETG_std
        obs = np.concatenate((obs, ETG_out), axis=0)

      if "ETG_obs" in self.sensor_mode.keys() and self.sensor_mode["ETG_obs"]:
        ETG_obs = info["ETG_obs"]
        obs = np.concatenate((obs, ETG_obs), axis=0)

    if "force_vec" in self.sensor_mode.keys() and self.sensor_mode["force_vec"]:
      force_vec = info["force_vec"]
      obs = np.concatenate((obs, force_vec), axis=0)

    if "dynamic_vec" in self.sensor_mode.keys() and self.sensor_mode["dynamic_vec"]:
      dynamic_vec = self.dynamic_info
      obs = np.concatenate((obs, dynamic_vec), axis=0)

    if "yaw" in self.sensor_mode.keys() and self.sensor_mode["yaw"]:
      if "d_yaw" in kwargs.keys():
        d_yaw = kwargs["d_yaw"]
      else:
        d_yaw = 0
      yaw_now = info["rot_euler"][-1]
      yaw_info = np.array([np.cos(d_yaw - yaw_now), np.sin(d_yaw - yaw_now)])
      obs = np.concatenate((obs, yaw_info), axis=0)

    if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"] > 0:
      self.obs_history = np.zeros((self.time_steps * self.time_interval, self.sensor_shape))
      obs_list = []
      for t in range(self.time_steps):
        obs_list.append(copy(self.obs_history[t * self.time_interval]))
      obs_list.append(copy(obs))
      self.obs_history[-1] = copy(obs)
      if self.sensor_mode["RNN"]["mode"] == "GRU":
        obs = np.stack(obs_list, axis=0)
      elif self.sensor_mode["RNN"]["mode"] == "stack":
        obs = np.array(obs_list).reshape(-1)

    return obs, info

  def step(self, action, **kwargs):
    obs, rew, done, info = self.env.step(action, **kwargs)
    if self.ETG:
      if "ETG" in self.sensor_mode.keys() and self.sensor_mode["ETG"]:
        ETG_out = info["ETG_act"]  # expanded policy input, 12
        if self.normal:
          ETG_out = (ETG_out - self.ETG_mean) / self.ETG_std
        obs = np.concatenate((obs, ETG_out), axis=0)

      if "ETG_obs" in self.sensor_mode.keys() and self.sensor_mode["ETG_obs"]:
        ETG_obs = info["ETG_obs"]  # 20
        obs = np.concatenate((obs, ETG_obs), axis=0)

    if "force_vec" in self.sensor_mode.keys() and self.sensor_mode["force_vec"]:
      force_vec = info["force_vec"]  # 6
      obs = np.concatenate((obs, force_vec), axis=0)

    if "dynamic_vec" in self.sensor_mode.keys() and self.sensor_mode["dynamic_vec"]:
      dynamic_vec = self.dynamic_info  # 3
      obs = np.concatenate((obs, dynamic_vec), axis=0)

    if "yaw" in self.sensor_mode.keys() and self.sensor_mode["yaw"]:
      if "d_yaw" in kwargs.keys():
        d_yaw = kwargs["d_yaw"]
      else:
        d_yaw = 0
      yaw_now = info["rot_euler"][-1]
      yaw_info = np.array([np.cos(d_yaw - yaw_now), np.sin(d_yaw - yaw_now)])
      obs = np.concatenate((obs, yaw_info), axis=0)

    if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"] > 0:
      obs_list = []
      for t in range(self.time_steps):
        obs_list.append(copy(self.obs_history[t * self.time_interval]))
      obs_list.append(copy(obs))
      self.obs_history[:-1] = copy(self.obs_history[1:])
      self.obs_history[-1] = copy(obs)
      if self.sensor_mode["RNN"]["mode"] == "GRU":
        obs = np.stack(obs_list, axis=0)
      elif self.sensor_mode["RNN"]["mode"] == "stack":
        obs = np.array(obs_list).reshape(-1)
    return obs, rew, done, info


class ETGWrapper(gym.Wrapper):
  def __init__(self, env, ETG, ETG_T, ETG_path, ETG_T2, ETG_H=20, act_mode="traj", task_mode="normal", step_y=0.05):
    gym.Wrapper.__init__(self, env)
    self.robot = self.env.robot
    self.pybullet_client = self.env.pybullet_client
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.ETG_T2 = ETG_T2
    self.ETG_T = ETG_T
    self.ETG_H = ETG_H
    self.act_mode = act_mode
    self.step_y = step_y
    self.task_mode = task_mode
    self.ETG = ETG
    phase = np.array([-np.pi / 2, 0])
    if self.ETG:
      self.ETG_agent = ETG_layer(self.ETG_T, self.env.env_time_step, self.ETG_H, 0.04, phase, 0.2, self.ETG_T2)
      self.ETG_weight = 1
      if len(ETG_path) > 1 and os.path.exists(ETG_path):
        info = np.load(ETG_path)
        self.ETG_w = info["w"]
        self.ETG_b = info["b"]
      else:
        self.ETG_w = np.zeros((3, ETG_H))
        self.ETG_b = np.zeros(3)
      self.ETG_model = ETG_model(task_mode=self.task_mode, act_mode=act_mode, step_y=self.step_y)
      self.last_ETG_act = np.zeros(12)
      self.last_ETG_obs = np.zeros(self.ETG_H)

  def reset(self, **kwargs):
    kwargs["info"] = True
    obs, info = self.env.reset(**kwargs)
    if self.ETG:
      if "ETG_w" in kwargs.keys() and kwargs["ETG_w"] is not None:
        self.ETG_w = kwargs["ETG_w"]
      if "ETG_b" in kwargs.keys() and kwargs["ETG_b"] is not None:
        self.ETG_b = kwargs["ETG_b"]
      self.ETG_agent.reset()
      state = self.ETG_agent.update2(t=self.env.get_time_since_reset())
      act_ref = self.ETG_model.forward(self.ETG_w, self.ETG_b, state)
      act_ref = self.ETG_model.act_clip(act_ref, self.robot)
      self.last_ETG_act = act_ref * self.ETG_weight
      info["ETG_obs"] = state[0]
      info["ETG_act"] = self.last_ETG_act
    return obs, info

  def step(self, action, **kwargs):
    if self.ETG:
      # residual_act(input) + ETG_act
      action = np.asarray(action).reshape(-1) + self.last_ETG_act
      # CPG-RBF, the output of the hidden neuron: V(t), [tuple:2(20x1)]
      t = self.env.get_time_since_reset()
      state = self.ETG_agent.update2(t)
      # P(t) = W ∗ V(t) + b, W:[3x20], b:[3x1] The phase difference of TG is T/2.
      act_ref = self.ETG_model.forward(self.ETG_w, self.ETG_b, state)
      # local position in foot link's frame
      action_before = act_ref
      ### Use IK to compute the motor angles, act_ref = etg_act - init_act !
      act_ref = self.ETG_model.act_clip(act_ref, self.robot)
      self.last_ETG_act = act_ref * self.ETG_weight
      obs, rew, done, info = self.env.step(action)
      # if abs(t % 0.5) < 1e-5:
      #   print('swing : 112233')
      # if abs(t % 0.5 - 0.25) < 1e-5:
      #   print('stance: 445566')
      info["ETG_obs"] = state[0]
      info["ETG_act"] = self.last_ETG_act
    else:
      obs, rew, done, info = self.env.step(action)
    return obs, rew, done, info


class RewardShaping(gym.Wrapper):
  def __init__(self, env, reward_param, reward_p=1, vel_d=0.6, vel_mode="max", foot_dx=0.2):
    gym.Wrapper.__init__(self, env)
    self.reward_param = reward_param
    self.reward_p = reward_p
    self.vel_d = vel_d
    self.vel_mode = vel_mode
    self.foot_dx = foot_dx
    self.last_base10 = np.zeros((10, 3))
    self.robot = self.env.robot
    self.pybullet_client = self.env.pybullet_client
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.steps = 0
    self.yaw_init = 0.0
    self.last_contacts = np.zeros(4)
    self.feet_air_time = np.zeros(4)
    self.last_action = np.zeros(12)

  def reset(self, **kwargs):
    self.steps = 0
    obs, info = self.env.reset(**kwargs)
    self.yaw_init = info["yaw_init"]
    obs, rew, done, infos = self.env.step(np.zeros(self.action_space.high.shape[0]))
    self.last_basepose = info["base_position"]
    self.last_footposition = self.get_foot_world(info)
    self.last_contact_position = self.last_footposition
    base_pose = info["base_position"]
    self.last_base10 = np.tile(base_pose, (10, 1))
    self.feetpos_err = []
    info["foot_position_world"] = copy(self.last_footposition)
    info["scene"] = "plane"
    if "d_yaw" in kwargs.keys():
      info["d_yaw"] = kwargs["d_yaw"]
    else:
      info["d_yaw"] = 0
    if self.render:
      self.line_id = self.draw_direction(info)
    # return obs, info
    return obs

  def step(self, action, **kwargs):
    self.steps += 1
    obs, rew, done, info = self.env.step(action, **kwargs)
    self.env_vec = np.array([0, 0, 0, 0, 0, 0, 0])
    posex = info["base_position"][0]
    for env_v in info["env_info"]:  # [lastx, basex, env_vec]
      if posex + 0.2 >= env_v[0] and posex + 0.2 <= env_v[1]:
        self.env_vec = env_v[2]
        break
    if self.env_vec[0]:
      info["scene"] = "upslope"
    elif self.env_vec[1]:
      info["scene"] = "downslope"
    elif self.env_vec[2]:
      info["scene"] = "upstair"
    elif self.env_vec[3]:
      info["scene"] = "downstair"
    else:
      info["scene"] = "plane"
    info["vel"] = (np.array(info["base_position"]) - np.array(self.last_basepose)) / self.env.env_time_step
    info["d_yaw"] = kwargs["d_yaw"] if "d_yaw" in kwargs.keys() else 0
    donef = kwargs["donef"] if "donef" in kwargs.keys() else False
    # info["foot_position_world"] = self.get_foot_world(info)
    # get reward terms, stored in the "info"
    info = self.reward_shaping(obs, rew, done, info, action, donef)
    rewards = 0
    done = self.terminate(info)
    info['rew_done'] = -1 * self.reward_param['rew_done'] if done else 0
    # sum the reward terms
    for key in Param_Dict.keys():
      if key in info.keys():
        # print(key)
        rewards += info[key]
    self.last_basepose = copy(info["base_position"])
    self.last_contacts = copy(info["real_contact"])
    self.last_base10[1:, :] = self.last_base10[:9, :]
    self.last_base10[0, :] = np.array(info["base_position"]).reshape(1, 3)
    if self.render:
      self.pybullet_client.removeUserDebugItem(self.line_id)
      self.line_id = self.draw_direction(info)
      self.draw_footstep(info)
    return (obs, self.reward_p * rewards, done, info)  # reward_p: default 5.0

  def reward_shaping(self, obs, rew, done, info, action, donef, last_basepose=None, last_footposition=None):
    if last_basepose is None:
      v = (np.array(info["base_position"]) - np.array(self.last_basepose)) / self.env.env_time_step
    else:
      v = (np.array(info["base_position"]) - np.array(last_basepose)) / self.env.env_time_step
    k = 1 - self.c_prec(min(v[0], self.vel_d), self.vel_d, 0.5)
    info['rew_torso'] = self.reward_param['rew_torso'] * self.re_torso(info, last_basepose=last_basepose)  # base velocity
    info['rew_up'] = self.reward_param['rew_up'] * self.re_up(info) * k  # roll and pitch
    info['rew_feet_vel'] = self.reward_param['rew_feet_vel'] * self.re_feet_velocity(info, last_footposition=last_footposition)  # foot velocity
    info['rew_feet_airtime'] = self.reward_param['rew_feet_airtime'] * self.re_feet_air_time(info)
    info['rew_feet_pos'] = self.reward_param['rew_feet_pos'] * self.re_feet_position(info)
    info['rew_velx'] = self.reward_param['rew_velx'] * rew  # calculated in "SimpleForwardTask": current_base_pos[0] - last_base_pos[0]
    info['rew_actionrate'] = self.reward_param['rew_actionrate'] * self.re_action_rate(action, info)
    # discouraged terms
    info['rew_tau'] = -self.reward_param['rew_tau'] * info["energy"] * k
    info['rew_badfoot'] = -self.reward_param['rew_badfoot'] * self.robot.GetBadFootContacts()
    lose_contact_num = np.sum(1.0 - np.array(info["real_contact"]))
    info['rew_footcontact'] = -self.reward_param['rew_footcontact'] * max(lose_contact_num - 2, 0)
    return info

  def draw_direction(self, info):
    pose = info["base_position"]
    if self.render:
      id = self.pybullet_client.addUserDebugLine(lineFromXYZ=[pose[0], pose[1], 0.6],
                                                 lineToXYZ=[pose[0] + np.cos(info['d_yaw']),
                                                            pose[1] + np.sin(info['d_yaw']), 0.6],
                                                 lineColorRGB=[1, 0, 1], lineWidth=2)
    return id

  def draw_footstep(self, info):
    color_list = [[1,1,0], [1,0,0], [0,0,1], [0,1,0]]
    self.pybullet_client.removeAllUserDebugItems()
    for i in range(self.last_contact_position.shape[0]):
      contact_pos = self.last_contact_position[i] + [self.foot_dx, 0, 0]
      if i % 2 == 1: y_pos = 0.135
      else: y_pos = -0.135
      z_pos = self.env.get_terrain_height(contact_pos[0], y_pos)
      self.pybullet_client.addUserDebugLine(lineFromXYZ=[contact_pos[0]-0.02, y_pos, z_pos],
                                            lineToXYZ=[contact_pos[0]+0.02, y_pos, z_pos],
                                            lineColorRGB=color_list[i], lineWidth=20)

  def terminate(self, info):
    rot_mat = info["rot_mat"]
    pose = info["rot_euler"]
    footposition = copy(info["foot_position"])
    footz = footposition[:, -1]
    base = info["base_position"]
    base_std = np.sum(np.std(self.last_base10, axis=0))
    return rot_mat[-1] < 0.5 or np.mean(footz) > -0.1 or np.max(footz) > 0 or (
          base_std <= 2e-4 and self.steps >= 10) or abs(pose[-1]) > 0.6

  def _calc_torque_reward(self):
    energy = self.robot.GetEnergyConsumptionPerControlStep()
    return -energy

  def re_still(self, info):
    v = (np.array(info["base_position"]) - np.array(self.last_basepose)) / self.env.env_time_step
    return -np.linalg.norm(v)

  def re_standupright(self, info):
    still = self.re_still(info)
    up = self.re_up(info)
    return self.re_rot(info, still + up)

  def re_up(self, info):
    posex = info["base_position"][0]
    env_vec = np.zeros(7)
    for env_v in info["env_info"]:
      if posex + 0.2 >= env_v[0] and posex + 0.2 <= env_v[1]:
        env_vec = env_v[2]
        break
    pose = copy(info["rot_euler"])
    roll = pose[0]
    pitch = pose[1]
    if env_vec[0]:
      pitch += abs(env_vec[4])
    elif env_vec[1]:
      pitch -= abs(env_vec[4])
    r = np.sqrt(roll ** 2 + pitch ** 2)
    return 1 - self.c_prec(r, 0, 0.4)

  def re_rot(self, info, r):
    pose = copy(info["rot_euler"])
    yaw = pose[-1]
    k1 = 1 - self.c_prec(yaw, info['d_yaw'], 0.5)
    k2 = 1 - self.c_prec(yaw, info['d_yaw'] + 2 * np.pi, 0.5)
    k3 = 1 - self.c_prec(yaw, info['d_yaw'] - 2 * np.pi, 0.5)
    k = max(k1, k2, k3)
    return min(k * r, r)

  def c_prec(self, v, t, m):
    # w = np.arctanh(np.sqrt(0.95)) / m  # 2.89 / m
    w = np.sqrt(np.arctanh(0.95)) / m  # 1.35 / m
    return np.tanh(np.power((v - t) * w, 2))

  def re_action_rate(self, action, info):
    new_action = np.array(info["real_action"])
    if self.last_action[0] == 0.0:
      self.last_action = new_action
      return 0.0;
    r = np.square(new_action - self.last_action)
    r = np.sum(r, axis=0)
    self.last_action = new_action
    return 1 - self.c_prec(r, 0, 0.2)

  def re_feet_velocity(self, info, vd=[1, 0, 0], last_footposition=None):
    vd[0] = np.cos(info['d_yaw'])
    vd[1] = np.sin(info['d_yaw'])
    posex = info["base_position"][0]
    env_vec = np.zeros(7)
    for env_v in info["env_info"]:
      if posex + 0.2 >= env_v[0] and posex + 0.2 <= env_v[1]:
        env_vec = env_v[2]
        break
    if env_vec[0]:
      vd[0] *= abs(np.cos(env_vec[4]))
      vd[1] *= abs(np.cos(env_vec[4]))
      vd[2] = abs(np.sin(env_vec[4]))
    elif env_vec[1]:
      vd[0] *= abs(np.cos(env_vec[4]))
      vd[1] *= abs(np.cos(env_vec[4]))
      vd[2] = -abs(np.sin(env_vec[4]))
    footposition = self.get_foot_world(info)
    if last_footposition is None:
      d_foot = (footposition - self.last_footposition) / self.env.env_time_step
    else:
      d_foot = (footposition - last_footposition) / self.env.env_time_step
    v_sum = 0
    contact = copy(info["real_contact"])
    for i in range(4):
      v = d_foot[i]
      v_ = v[0] * vd[0] + v[1] * vd[1] + v[2] * vd[2]
      r = min(v_, self.vel_d) / 4.0
      v_sum += min(r, 1.0 * r)
    return self.re_rot(info, v_sum)

  def re_feet_air_time(self, info):
    # Reward long steps
    contact = np.array(copy(info["real_contact"]))
    contact_filt = np.logical_or(contact, np.array(self.last_contacts))
    first_contact = (self.feet_air_time > 0.) * contact_filt
    self.feet_air_time += self.env.env_time_step
    # if first_contact[0] == True:
    #   print(self.feet_air_time)
    rew_airtime = np.sum((self.feet_air_time - 0.25) * first_contact, axis=0)
    self.feet_air_time *= ~contact_filt  # if last and current foot are both in air
    return rew_airtime
  
  def re_feet_position(self, info):
    # reward target foot position
    contact = np.array(info["real_contact"])
    contact_position = np.array(info["foot_position_world"])
    lift_contact = np.logical_xor(np.logical_or(self.last_contacts, contact), contact)
    first_contact = np.logical_xor(np.logical_or(self.last_contacts, contact), self.last_contacts)
    first_contact = np.repeat(first_contact.reshape(-1,1), 3, axis=1)
    feet_fly_length = (contact_position - self.last_contact_position) * first_contact
    rew_feet_length = 0
    for i in range(first_contact.shape[0]):
      if first_contact[i][0] == True:
        # feet_length_err = np.sum(np.sqrt(np.square(feet_fly_length[i] - np.array([0.3, 0.0, 0.0]))))
        foot_x, foot_y, foot_z = contact_position[i][0], contact_position[i][1], contact_position[i][2]
        world_z = self.env.get_terrain_height(foot_x, foot_y)
        feet_length_err = np.sqrt(np.sum(np.square(
          [feet_fly_length[i][0]-self.foot_dx, abs(foot_y)-0.135, foot_z-world_z])))
        # rew_feet_length += feet_length_err
        rew_feet_length += (-self.c_prec(feet_length_err, 0, 0.05))
        self.feetpos_err.append(feet_length_err)
        # print("Average feetpos error: ", np.mean(self.feetpos_err))
        # if i == 0:
        #   print(feet_length_err)
        self.last_contact_position[i] = contact_position[i]
      # if lift_contact[i] == True:
      #   self.last_contact_position[i] = contact_position[i]
    # rew_feet_length = 1 - self.c_prec(rew_feet_length, 0, 0.1)  # positive reward
    # rew_feet_length = -self.c_prec(rew_feet_length, 0, 0.1)  # negative penalty
    return rew_feet_length

  def get_foot_world(self, info={}):
    if "foot_position" in info.keys():
      foot = np.array(info["foot_position"]).transpose()
      rot_mat = np.array(info["rot_mat"]).reshape(-1, 3)
      base = np.array(info["base_position"]).reshape(-1, 1)
    else:
      foot = np.array(self.robot.GetFootPositionsInBaseFrame()).transpose()
      rot_quat = self.robot.GetBaseOrientation()
      rot_mat = np.array(self.pybullet_client.getMatrixFromQuaternion(rot_quat)).reshape(-1, 3)
      base = np.array(self.robot.GetBasePosition()).reshape(-1, 1)
      print("no!")
    foot_world = rot_mat.dot(foot) + base
    return foot_world.transpose()

  def re_torso(self, info, vd=[1, 0, 0], last_basepose=None):
    if last_basepose is None:
      v = (np.array(info["base_position"]) - np.array(self.last_basepose)) / self.env.env_time_step
    else:
      v = (np.array(info["base_position"]) - np.array(last_basepose)) / self.env.env_time_step
    vd[0] = np.cos(info['d_yaw'])
    vd[1] = np.sin(info['d_yaw'])
    posex = info["base_position"][0]
    env_vec = np.zeros(7)
    for env_v in info["env_info"]:
      if posex + 0.2 >= env_v[0] and posex + 0.2 <= env_v[1]:
        env_vec = env_v[2]
        break
    if env_vec[0]:
      vd[0] *= abs(np.cos(env_vec[4]))
      vd[1] *= abs(np.cos(env_vec[4]))
      vd[2] = abs(np.sin(env_vec[4]))
    elif env_vec[1]:
      vd[0] *= abs(np.cos(env_vec[4]))
      vd[1] *= abs(np.cos(env_vec[4]))
      vd[2] = -abs(np.sin(env_vec[4]))
    if self.vel_mode == "max":
      v_ = v[0] * vd[0] + v[1] * vd[1] + v[2] * vd[2]
      v_reward = min(self.vel_d, v_)
    elif self.vel_mode == "equal":
      v_ = v[0] * vd[0] + v[1] * vd[1] + v[2] * vd[2]
      v_diff = abs(v_ - self.vel_d)
      v_reward = np.exp(-5 * v_diff)
    return self.re_rot(info, v_reward)


class RandomWrapper(gym.Wrapper):
  def __init__(self, env, random_param):
    gym.Wrapper.__init__(self, env)
    self.random_param = random_param if random_param is not None else {}
    self.robot = self.env.robot
    self.pybullet_client = self.env.pybullet_client
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.render = self.env.rendering_enabled

  def generate_randomforce(self):
    force_position = (np.random.random(3) - 0.5) * 2 * np.array([0.2, 0.05, 0.05])
    force_vec = np.random.uniform(low=-1, high=1, size=(3,)) * np.array([0.5, 1, 0.05])
    force_vec = force_vec / np.linalg.norm(force_vec) * np.random.uniform(20, 50)
    return force_position, force_vec

  def draw_forceline(self, force_position, force_vec):
    if self.render:
      self.pybullet_client.addUserDebugLine(lineFromXYZ=force_position, lineToXYZ=force_position + force_vec / 50,
                                            parentObjectUniqueId=self.robot.quadruped,
                                            parentLinkIndex=-1, lineColorRGB=[1, 0, 0])

  def random_dynamics(self, info):
    footfriction = 1
    footfriction_normal = 0

    basemass = self.robot.GetBaseMassesFromURDF()[0]
    basemass_ratio_normal = 0

    baseinertia = self.robot.GetBaseInertiasFromURDF()
    baseinertia_ratio_normal = np.zeros(3)

    legmass = self.robot.GetLegMassesFromURDF()
    legmass_ratio_normal = np.zeros(3)

    leginertia = self.robot.GetLegInertiasFromURDF()
    leginertia_ratio_normal = np.zeros(3)

    control_latency = 0
    control_latency_normal = -1

    joint_friction = 0.025
    joint_friction_normal = [0]
    joint_friction_vec = np.array([joint_friction] * 12)

    spin_friction = 0.2
    spin_friction_normal = 0

    if "random_dynamics" in self.random_param.keys() and self.random_param["random_dynamics"]:
      # friction
      # footfriction = np.random.uniform(1,2.5)
      # footfriction_normal = footfriction-1

      # basemass
      # basemass_ratio = np.random.uniform(0.8,1.2)
      # basemass_ratio_normal = (basemass_ratio-1)/0.2
      # basemass = basemass*basemass_ratio

      # baseinertia
      # baseinertia_ratio = np.random.uniform(0.8,1.2,3)
      # baseinertia_ratio_normal = (baseinertia_ratio-1)/0.2
      # baseinertia = baseinertia[0]
      # baseinertia = [(baseinertia[0]*baseinertia_ratio[0],baseinertia[1]*baseinertia_ratio[1],baseinertia[2]*baseinertia_ratio[2])]

      # legmass
      # legmass_ratio = np.random.uniform(0.8,1.2,3)
      # legmass_ratio_normal = (legmass_ratio-1)/0.2
      # legmass = legmass*np.array([legmass_ratio[0],legmass_ratio[1],legmass_ratio[2]]*4)

      # leginertia
      # leginertia_ratio = np.random.uniform(0.8,1.2,3)
      # leginertia_ratio_normal = (leginertia_ratio-1)/0.2
      # leginertia_new = []
      # for lg in leginertia:
      #      leginertia_new.append(leginertia_ratio*lg)
      # leginertia = copy(leginertia_new)

      # #control_latency
      control_latency = np.random.uniform(0.01, 0.02)
      control_latency_normal = (control_latency - 0.01) / 0.01
      print("latency:", control_latency)
      # joint_friction
      # joint_friction = np.random.random(1)*0.05
      # joint_friction_normal = (joint_friction/0.05-0.5)*2
      # joint_friction_vec = np.array([joint_friction]*12)

      # spin_friction
      # spin_friction = np.random.uniform(0,0.4)
      # spin_friction_normal = (spin_friction-0.2)*5

    dynamics_vec = np.concatenate(([footfriction_normal], [basemass_ratio_normal],
                                   baseinertia_ratio_normal, legmass_ratio_normal,
                                   leginertia_ratio_normal, [control_latency_normal],
                                   joint_friction_normal, [spin_friction_normal]), axis=0)
    self.robot.SetFootFriction(footfriction)
    self.robot.SetBaseMasses([basemass])
    self.robot.SetBaseInertias(baseinertia)
    self.robot.SetLegMasses(legmass)
    self.robot.SetLegInertias(leginertia)
    self.robot.SetControlLatency(control_latency)
    self.robot.SetJointFriction(joint_friction_vec)
    self.robot.SetFootSpinFriction(spin_friction)
    info['dynamics'] = dynamics_vec
    return info

  def reset(self, **kwargs):
    # infos = self.random_dynamics({})
    obs, info = self.env.reset(**kwargs)
    info['dynamics'] = np.array([info['latency'], info["footfriction"], info['basemass']])
    force_info = np.zeros(6)
    if "random_force" in self.random_param.keys() and self.random_param["random_force"]:
      self.pybullet_client.removeAllUserDebugItems()
      self.force_position, self.force_vec = self.generate_randomforce()
      self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped, linkIndex=-1,
                                              forceObj=self.force_vec,
                                              posObj=self.force_position, flags=self.pybullet_client.LINK_FRAME)
      self.draw_forceline(self.force_position, self.force_vec)
      force_info = np.concatenate((self.force_position / np.array([0.2, 0.05, 0.05]), self.force_vec / 50), axis=0)
    # info = self.random_dynamics(info)
    info['force_vec'] = force_info
    # print("init_info",info)
    return obs, info

  def step(self, action, **kwargs):
    force_info = np.zeros(6)
    obs, rew, done, info = self.env.step(action, **kwargs)
    if "random_force" in self.random_param.keys() and self.random_param["random_force"]:
      # New random force
      if self.env.env_step_counter % 100 == 0:
        self.force_position, self.force_vec = self.generate_randomforce()
        self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped, linkIndex=-1,
                                                forceObj=self.force_vec,
                                                posObj=self.force_position, flags=self.pybullet_client.LINK_FRAME)
        self.draw_forceline(self.force_position, self.force_vec)
        force_info = np.concatenate((self.force_position / np.array([0.2, 0.05, 0.05]), self.force_vec / 50), axis=0)
      # Apply force
      elif self.env.env_step_counter % 100 < 50:
        self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped, linkIndex=-1,
                                                forceObj=self.force_vec,
                                                posObj=self.force_position, flags=self.pybullet_client.LINK_FRAME)
        self.draw_forceline(self.force_position, self.force_vec)
        force_info = np.concatenate((self.force_position / np.array([0.2, 0.05, 0.05]), self.force_vec / 50), axis=0)
      # delete line
      elif self.env.env_step_counter % 100 == 50:
        self.pybullet_client.removeAllUserDebugItems()
    info['force_vec'] = force_info
    return obs, rew, done, info
