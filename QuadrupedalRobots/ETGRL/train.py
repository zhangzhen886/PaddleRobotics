#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import time
import datetime
from copy import copy, deepcopy
# from concurrent.futures import ThreadPoolExecutor
# from queue import Queue
import numpy as np
import gym
import pybullet as p
import cv2
import argparse
import logging
from colorlog import ColoredFormatter
import json
from matplotlib import pyplot as plt
import wandb

from parl.utils import summary, ReplayMemory
from model.mujoco_model import MujocoModel
from model.mujoco_agent import MujocoAgent
from alg.sac import SAC
from alg.es import SimpleGA
import rlschool.quadrupedal as quadrupedal
from rlschool.quadrupedal.envs.utilities.ETG_model import ETG_layer, ETG_model
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import Param_Dict, Random_Param_Dict, Info_Dict
from rlschool.quadrupedal.envs.env_builder import SENSOR_MODE
from rlschool.quadrupedal.robots import robot_config
import ma_env_wrappers


WARMUP_STEPS = 1e4
EVAL_EVERY_STEPS = 1e4
ES_EVERY_STEPS = 5e4
ES_TRAIN_STEPS = 10
# PARTICLE_NUM = 100
# PER_EPISODE = 3
# EVAL_EPISODES = 1
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
STEP_HEIGHT = np.arange(0.08, 0.101, 0.002)
SLOPE = np.arange(0.2, 0.401, 0.02)
STEP_WIDTH = np.arange(0.26, 0.401, 0.02)
default_pose = np.array([0, 0.9, -1.8] * 4)
ENV_NUMS = 8

random_param = copy(Random_Param_Dict)
reward_param = copy(Param_Dict)
sensor_param = copy(SENSOR_MODE)
mode_map = {"pose"  : robot_config.MotorControlMode.POSITION,
            "torque": robot_config.MotorControlMode.TORQUE,
            "traj"  : robot_config.MotorControlMode.POSITION, }

plt.rcParams['figure.dpi'] = 300
logger = logging.getLogger(__name__)

def plot_gait(w, b, ETG, points, save_path=None, show_fig=True):
  w = np.vstack((w[0,], w[-1,]))
  b = np.hstack((b[0], b[-1]))
  t_t = np.arange(0.0, ETG.T, ETG.T / 100)
  p = []
  for t in np.nditer(t_t):
    p.append(w.dot(ETG.update(t)) + b)
  p = np.array(p)
  colors = t_t * 200

  fig0 = plt.figure()
  plt.scatter(points[:, 0], points[:, 1], c='red', alpha=0.5)
  plt.scatter(p[50, 0], p[50, 1], c='blue', alpha=1)
  plt.scatter(p[0, 0], p[0, 1], c='blue', alpha=1)
  plt.scatter(p[:, 0], p[:, 1], s=10, c=colors, cmap='viridis')
  plt.xlim(-0.15, 0.15)
  plt.ylim(-0.10, 0.15)
  plt.tight_layout()
  # plt.colorbar()
  fig1, axs = plt.subplots(2, 1, sharex=True)
  plt.xlim(0.0, ETG.T)
  plt.xlabel("time(s)")
  axs[0].set_ylim(-0.10, 0.10)
  axs[0].set_ylabel("x(m)")
  axs[0].scatter(t_t, p[:, 0], s=10, c='blue')
  axs[1].set_ylim(-0.03, 0.13)
  axs[1].set_ylabel("z(m)")
  axs[1].scatter(t_t, p[:, 1], s=10, c='blue')
  plt.tight_layout()
  if save_path is not None:
    plot_path = save_path + '_1.png'
    fig0.savefig(plot_path)
    plot_path = save_path + '_2.png'
    fig1.savefig(plot_path)
    if show_fig is True:
      plt.show()
    plt.close(fig0)
    plt.close(fig1)
  else:
    plt.show()


# solve "x" of "Ax = b"
def LS_sol(A, b, precision=1e-4, alpha=0.05, lamb=1, w0=None):
  n, m = A.shape  # 6x20
  if w0 is not None:
    x = copy(w0)
  else:
    x = np.zeros((m, 1))  # 20x1
  err = A.dot(x) - b  # 6x1
  err = err.transpose().dot(err)
  i = 0
  diff = 1
  while err > precision and i < 1000:
    A1 = A.transpose().dot(A)  # 20x20
    dx = A1.dot(x) - A.transpose().dot(b)
    if w0 is not None:
      dx += lamb * (x - w0)
    x = x - alpha * dx
    diff = np.linalg.norm(dx)
    err = A.dot(x) - b
    err = err.transpose().dot(err)
    i += 1
  return x


def Opt_with_points(ETG, ETG_T=0.4, ETG_H=20, points=None, b0=None, w0=None, precision=1e-4, lamb=0.5, **kwargs):
  ts = [0.5 * ETG_T + 0.1, 0, 0.05, 0.1, 0.15, 0.2]
  if points is None:
    Steplength = kwargs["Steplength"] if "Steplength" in kwargs else 0.05
    Footheight = kwargs["Footheight"] if "Footheight" in kwargs else 0.08
    Penetration = kwargs["Penetration"] if "Penetration" in kwargs else 0.01
    # [[0.0, -0.01], [-0.05, -0.005], [-0.075, 0.06], [0.0, 0.1], [0.075, 0.06], [0.05, -0.005]]
    points = np.array([[0, -Penetration],
                       [-Steplength, -Penetration * 0.5], [-Steplength * 1.5, 0.6 * Footheight],
                       [0, Footheight],
                       [Steplength * 1.5, 0.6 * Footheight], [Steplength, -Penetration * 0.5]])
  obs = []
  for t in ts:
    v = ETG.update(t)  # calculate V(t), 20 dim
    obs.append(v)
  obs = np.array(obs).reshape(-1, ETG_H)  # V(1-6), 6x(20x1)
  if b0 is None:
    b = np.mean(points, axis=0)
  else:
    b = np.array([b0[0], b0[-1]])  # 2x1
  points_t = points - b  # 6x(2x1), W*V(t)=P(t)-b
  if w0 is None:
    x1 = LS_sol(A=obs, b=points_t[:, 0].reshape(-1, 1), precision=precision, alpha=0.05)  # 1x20, "x" axis
    x2 = LS_sol(A=obs, b=points_t[:, 1].reshape(-1, 1), precision=precision, alpha=0.05)  # 1x20, "z" axis
  else:
    x1 = LS_sol(A=obs, b=points_t[:, 0].reshape(-1, 1), precision=precision, alpha=0.05, lamb=lamb,
                w0=w0[0, :].reshape(-1, 1))
    x2 = LS_sol(A=obs, b=points_t[:, 1].reshape(-1, 1), precision=precision, alpha=0.05, lamb=lamb,
                w0=w0[-1, :].reshape(-1, 1))
  # x1 = np.zeros((20,1))
  w_ = np.stack((x1, np.zeros((ETG_H, 1)), x2), axis=0).reshape(3, -1)  # 3x20
  b_ = np.array([b[0], 0, b[1]])  # 3x1
  return w_, b_, points


def param2dynamic_dict(params):
  param = copy(params)
  param = np.clip(param, -1, 1)
  dynamic_param = {}
  dynamic_param['control_latency'] = np.clip(40 + 10 * param[0], 0, 80)
  dynamic_param['footfriction'] = np.clip(0.2 + 10 * param[1], 0, 20)
  dynamic_param['basemass'] = np.clip(1.5 + 1 * param[2], 0.5, 3)
  dynamic_param['baseinertia'] = np.clip(
    np.ones(3) + 1 * param[3:6], np.array([0.1] * 3), np.array([3] * 3)).tolist()
  dynamic_param['legmass'] = np.clip(
    np.ones(3) + 1 * param[6:9], np.array([0.1] * 3), np.array([3] * 3)).tolist()
  dynamic_param['leginertia'] = np.clip(
    np.ones(12) + 1 * param[9:21], np.array([0.1] * 12), np.array([3] * 12)).tolist()
  dynamic_param['motor_kp'] = np.clip(
    80 * np.ones(12) + 40 * param[21:33], np.array([20] * 12), np.array([200] * 12)).tolist()
  dynamic_param['motor_kd'] = np.clip(
    np.array([1., 2., 2.] * 4) + param[33:45] * np.array([1, 2, 2] * 4), np.array([0] * 12), np.array([5] * 12)).tolist()
  if param.shape[0] > 45:
    dynamic_param['gravity'] = np.clip(
      np.array([0, 0, -10]) + param[45:48] * np.array([2, 2, 10]), np.array([-5, -5, -20]), np.array([5, 5, -4])).tolist()
  return dynamic_param


# Run episode for training
def run_train_episode(agent, env, rpm, max_step, action_bound, w=None, b=None):
  action_dim = env.action_space.shape[0]
  obs = env.reset(ETG_w=w, ETG_b=b, x_noise=args.x_noise, heightfield_terrain=args.hft)
  terminal = False
  episode_reward, episode_steps = 0, 0
  infos = {}
  success_num = 0
  critic_loss_list = []
  actor_loss_list = []
  sample_time = 0.0
  step_time = 0.0
  rpm_time = 0.0
  learn_time = 0.0
  while True:
    episode_steps += 1
    t0 = time.time()
    # Select action randomly or according to policy, WARMUP_STEPS==1e4
    actions = []
    new_actions = []
    for i in range(env.num_envs):
      if rpm.size() < WARMUP_STEPS:
        action = np.random.uniform(-1, 1, size=action_dim)
      else:
        action = agent.sample(obs[i])
      actions.append(copy(action))
      new_actions.append(copy(action * action_bound))
      # if len(new_action) == 0:
      #   new_action = copy(action * action_bound)
      # else:
      #   new_action = np.vstack((new_action, action * action_bound))
    t1 = time.time()
    sample_time += t1 - t0
    # Perform action, action_bound: [0.3,0.3,0.3] * 4
    # next_obs, reward, done, info = env.step(new_action * action_bound, donef=(episode_steps > max_step))
    next_obs, reward, done, info = env.step(new_actions)
    t2 = time.time()
    step_time += t2 - t1
    for i in range(env.num_envs):
      terminal = float(done[i]) if episode_steps < 2000 else 0
      terminal = 1. - terminal
      # Store data in replay memory
      rpm.append(obs[i], actions[i], reward[i], next_obs[i], terminal)
    for key in Param_Dict.keys():
      for i in range(env.num_envs):
        if key in info[i].keys():
          if key not in infos.keys():
            infos[key] = info[i][key] / env.num_envs
          else:
            infos[key] += info[i][key] / env.num_envs
    # if info["rew_velx"] >= 0.3:
    #   success_num += 1
    obs = next_obs
    episode_reward += np.sum(reward) / env.num_envs
    if episode_steps > max_step:
      break
  # Train agent after collecting sufficient data, off-policy actor-critic algorithm
  if rpm.size() >= BATCH_SIZE:
    for _ in range(int(rpm.size() * args.learn / BATCH_SIZE)):
      # critic_loss, actor_loss = agent.learn(rpm.sample_batch(BATCH_SIZE)), BATCH_SIZE=256
      batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
      t3 = time.time()
      critic_loss, actor_loss = agent.learn(  # MujocoAgent
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)
      learn_time += time.time() - t3
      critic_loss_list.append(critic_loss)
      actor_loss_list.append(actor_loss)
  if len(critic_loss_list) > 0:
    infos["critic_loss"] = np.mean(critic_loss_list)
    infos["actor_loss"] = np.mean(actor_loss_list)
  # infos["success_rate"] = success_num / episode_steps
  # logger.info('Torso:{} Feet:{} Up:{} Tau:{}'.format(infos['torso'],infos['feet'],infos['up'],infos['tau']))
  logger.info("[TrainTime] rpm_size: {} sample: {:.4f}, step: {:.4f}, learn: {:.4f}".format(
    rpm.size(), sample_time, step_time, learn_time))
  return episode_reward, episode_steps, infos

def run_train_episode1(agent, env, rpm, max_step, action_bound, w=None, b=None):
  action_dim = env.action_space.shape[0]
  obs = env.reset(ETG_w=w, ETG_b=b, x_noise=args.x_noise)
  done = False
  episode_reward, episode_steps = 0, 0
  infos = {}
  success_num = 0
  critic_loss_list = []
  actor_loss_list = []
  while not done:
    episode_steps += 1
    # Select action randomly or according to policy, WARMUP_STEPS==1e4
    if rpm.size() < WARMUP_STEPS:
      action = np.random.uniform(-1, 1, size=action_dim)
    else:
      action = agent.sample(obs)
    # action = np.zeros(action_dim)
    new_action = copy(action)  # initial residual control signal
    # Perform action, action_bound: [0.3,0.3,0.3] * 4
    next_obs, reward, done, info = env.step(new_action * action_bound)
    terminal = float(done) if episode_steps < 2000 else 0
    terminal = 1. - terminal
    # Store data in replay memory
    rpm.append(obs, action, reward, next_obs, terminal)
    for key in Param_Dict.keys():
      if key in info.keys():
        if key not in infos.keys():
          infos[key] = info[key]
        else:
          infos[key] += info[key]
    # if info["rew_velx"] >= 0.3:
    #   success_num += 1
    obs = next_obs
    episode_reward += reward
    if episode_steps > max_step:
      break
  # Train agent after collecting sufficient data, off-policy actor-critic algorithm
  if rpm.size() >= WARMUP_STEPS:
    # critic_loss, actor_loss = agent.learn(rpm.sample_batch(BATCH_SIZE)), BATCH_SIZE=256
    for _ in range(int(rpm.size() * args.learn / BATCH_SIZE)):
      batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
      critic_loss, actor_loss = agent.learn(
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)
      critic_loss_list.append(critic_loss)
      actor_loss_list.append(actor_loss)
  if len(critic_loss_list) > 0:
    infos["critic_loss"] = np.mean(critic_loss_list)
    infos["actor_loss"] = np.mean(actor_loss_list)
  # infos["success_rate"] = success_num / episode_steps
  # logger.info('Torso:{} Feet:{} Up:{} Tau:{}'.format(infos['torso'],infos['feet'],infos['up'],infos['tau']))
  return episode_reward, episode_steps, infos

# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, max_step, action_bound, w=None, b=None, pub_joint=None):
  avg_reward = 0.
  infos = {}
  steps_all = 0
  # obs, info = env.reset(ETG_w=w, ETG_b=b, x_noise=args.x_noise)
  obs = env.reset(ETG_w=w, ETG_b=b, x_noise=args.x_noise, heightfield_terrain=args.hft)
  done = False
  steps = 0
  step_time_all = 0.0
  plt.ion()
  plt.figure(1)
  plt.ylim(-0.3, -0.1)
  plt.xlim(0.05, 0.3)
  # while True:
  while not done:
    steps += 1
    t0 = time.time()
    action = agent.predict(obs)  # NN output: residual control signal
    t1 = time.time()
    pred_time = t1 - t0
    # print("pred time:", pred_time)
    new_action = copy(action)
    # new_action = np.zeros(12)
    obs, reward, done, info = env.step(new_action * action_bound)
    if args.eval == 1:
      img = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      cv2.imwrite("img/img{}.jpg".format(steps), img)
    t2 = time.time()
    step_time = t2 - t1
    step_time_all += step_time
    avg_reward += reward
    # print("step time:", t2)
    if pub_joint is not None:
      import rospy
      from sensor_msgs.msg import JointState
      joint_msg = JointState()
      joint_msg.name.append("torso_to_abduct_fr_j")
      joint_msg.name.append("abduct_fr_to_thigh_fr_j")
      joint_msg.name.append("thigh_fr_to_knee_fr_j")
      joint_msg.name.append("torso_to_abduct_fl_j")
      joint_msg.name.append("abduct_fl_to_thigh_fl_j")
      joint_msg.name.append("thigh_fl_to_knee_fl_j")
      joint_msg.name.append("torso_to_abduct_hr_j")
      joint_msg.name.append("abduct_hr_to_thigh_hr_j")
      joint_msg.name.append("thigh_hr_to_knee_hr_j")
      joint_msg.name.append("torso_to_abduct_hl_j")
      joint_msg.name.append("abduct_hl_to_thigh_hl_j")
      joint_msg.name.append("thigh_hl_to_knee_hl_j")
      for i in range(0, 12):
        joint_msg.position.append(info['real_action'][i])  # target joint pos (action)
      for i in range(0, 12):
        # joint_msg.position.append((info['ETG_act'] + info['init_act'])[i])  #
        joint_msg.position.append(info['joint_angle'][i])  # current joint pos
      joint_msg.header.stamp = rospy.Time.now()
      joint_msg.header.frame_id = "robot"
      pub_joint.publish(joint_msg)
      # plt.ylim(0.0, 0.1)
      # if info["real_contact"][0] == True:
      #   plt.scatter(info["foot_position_world"][0][0], info["foot_position_world"][0][2], s=5, c='red')
      # else:
      #   plt.scatter(info["foot_position_world"][0][0], info["foot_position_world"][0][2], s=5, c='blue')
      # plt.ylim(-0.3, -0.1)
      # plt.xlim(0.05, 0.3)
      # plt.scatter(info["ETG_trj"][0], info["ETG_trj"][2], s=5, c='red')
      # plt.scatter(info["foot_position"][0][0], info["foot_position"][0][2], s=5, c='blue')
      # plt.pause(0.0001)
    for key in Param_Dict.keys():
      if key in info.keys():
        if key not in infos.keys():
          infos[key] = info[key]
        else:
          infos[key] += info[key]
    if steps > max_step:
      break
  steps_all += steps
  plt.ioff()
  plt.close(1)
  # print("\033[1;32m[Evaluation] Average step time: {} step/second.\033[0m".format(steps_all/step_time_all))
  logger.debug("[Evaluation] Average step time: {} step/second.".format(steps_all/step_time_all))
  return avg_reward, steps_all, infos


def run_EStrain_episode(agent, env, max_step, action_bound, w=None, b=None, rpm=None):
  obs = env.reset(ETG_w=w, ETG_b=b, x_noise=args.x_noise, heightfield_terrain=args.hft)
  done = False
  episode_reward, episode_steps = 0, 0
  infos = {}
  success_num = 0
  while not done:
    episode_steps += 1
    # Select action randomly or according to policy
    # use RL policy(SAC) to generate action residual?
    action = agent.predict(obs)
    new_action = copy(action)
    # Perform action
    next_obs, reward, done, info = env.step(new_action * action_bound)
    terminal = float(done) if episode_steps < 2000 else 0
    terminal = 1. - terminal
    for key in Param_Dict.keys():
      if key in info.keys():
        if key not in infos.keys():
          infos[key] = info[key]
        else:
          infos[key] += info[key]
    if info["rew_velx"] >= 0.3:
      success_num += 1
    # Store data in replay memory
    if args.es_rpm and rpm != None:
      rpm.append(obs, action, reward, next_obs, terminal)
    obs = next_obs
    episode_reward += reward
    if episode_steps > max_step:
      break
  infos["success_rate"] = success_num / episode_steps
  # logger.info('Torso:{} Feet:{} Up:{} Tau:{}'.format(infos['torso'],infos['feet'],infos['up'],infos['tau']))
  # print("success_rate:",success_num/episode_steps)
  return episode_reward, episode_steps, infos


def main():
  ## ETG init
  phase = np.array([-np.pi / 2, 0])
  dt = args.action_repeat_num * 0.002  # default: 13 * 0.002 = 0.026s
  # args.ETG_T, args.ETG_H = 0.4, 20
  ETG_agent = ETG_layer(args.ETG_T, dt, args.ETG_H, 0.04, phase, 0.2, args.ETG_T2)
  # prior_points: 6x2 [[0.0, -0.01], [-0.05, -0.005], [-0.075, 0.06], [0.0, 0.1], [0.075, 0.06], [0.05, -0.005]]
  w0, b0, prior_points = Opt_with_points(ETG=ETG_agent, ETG_T=args.ETG_T, ETG_H=args.ETG_H,
                                         Footheight=args.footheight, Steplength=args.steplen)
  # plot_gait(w0, b0, ETG_agent, prior_points)
  # if not os.path.exists(ETG_path):
  #   np.savez(ETG_path, w=w0, b=b0, param=prior_points)

  dynamic_param = np.load("data/sigma0.5_exp0_dynamic_param9027.npy")
  dynamic_param = param2dynamic_dict(dynamic_param)

  ## create gym envs of the quadruped robot
  render = True if (args.eval or args.render) else False
  if args.eval == 0:
    env = []
    if args.env_nums != 0:
      ENV_NUMS = args.env_nums
    for _ in range(ENV_NUMS):
      a1_env = quadrupedal.A1GymEnv(task=args.task_mode, hf_terrain_mode=args.hft,
                                    motor_control_mode=mode_map[args.act_mode],
                                    render=render, on_rack=False, sensor_mode=sensor_param, normal=args.normal,
                                    reward_param=reward_param, random_param=random_param, dynamic_param=dynamic_param,
                                    ETG=args.ETG, ETG_T=args.ETG_T, ETG_H=args.ETG_H,
                                    vel_d=args.vel_d, foot_dx=args.foot_dx, step_y=args.step_y, reward_p=args.reward_p,
                                    enable_action_filter=args.enable_action_filter,
                                    action_repeat=args.action_repeat_num)
      env.append(a1_env)
    multi_env = ma_env_wrappers.SubprocVecEnv(env)
    obs_dim = multi_env.observation_space.shape[0]
    action_dim = multi_env.action_space.shape[0]
  # else:
  env = quadrupedal.A1GymEnv(task=args.task_mode, hf_terrain_mode=args.hft, motor_control_mode=mode_map[args.act_mode],
                             render=render, on_rack=False, sensor_mode=sensor_param, normal=args.normal,
                             reward_param=reward_param, random_param=random_param, dynamic_param=dynamic_param,
                             ETG=args.ETG, ETG_T=args.ETG_T, ETG_H=args.ETG_H,
                             vel_d=args.vel_d, foot_dx=args.foot_dx, step_y=args.step_y, reward_p=args.reward_p,
                             enable_action_filter=args.enable_action_filter, action_repeat=args.action_repeat_num)
  obs_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  rpm = ReplayMemory(max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

  if args.act_mode == "pose":
    act_bound = np.array([0.1, 0.7, 0.7] * 4)
  elif args.act_mode == "torque":
    act_bound = np.array([10] * 12)
  elif args.act_mode == "traj":
    act_bound_now = args.act_bound
    act_bound = np.array([act_bound_now, act_bound_now, act_bound_now] * 4)  # 0.3*12

  # Initialize RL model, algorithm, agent, replay_memory
  model = MujocoModel(obs_dim, action_dim)
  algorithm = SAC(
    model,
    gamma=GAMMA,
    tau=TAU,
    alpha=ALPHA,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR)
  RL_agent = MujocoAgent(algorithm)
  if args.resume != "":  # resume mode
    RL_agent.restore(args.resume + ".pt")
  elif args.load != "":
    RL_agent.restore(args.load)

  # training mode
  if not args.eval:
    if not os.path.exists(args.outdir):
      os.makedirs(args.outdir)
    today = datetime.date.today()
    formatted_today = today.strftime('%m%d')
    suffix = formatted_today + args.suffix + '_' + args.task_mode
    outdir = os.path.join(args.outdir, suffix)
    ## ES init
    ETG_path = ''
    ETG_param_init = np.zeros(12)  # dim: 6(points)*2(x&y)
    ETG_info = np.load("data/zero_param.npz")
    if args.resume != "":  # resume mode
      ETG_path = args.resume + ".npz"
    elif args.ETG_path != "":
      ETG_path = args.ETG_path
    if os.path.exists(ETG_path):
      ETG_info = np.load(ETG_path)
      ETG_param_init = ETG_info["param"].reshape(-1)
    ES_solver = SimpleGA(ETG_param_init.shape[0],
                         sigma_init=args.sigma,
                         sigma_decay=args.sigma_decay,
                         sigma_limit=0.005,
                         elite_ratio=0.1,
                         weight_decay=0.005,
                         popsize=args.popsize,
                         param=ETG_param_init)

    # dump the args and params as files
    if not os.path.exists(outdir) and args.resume == "":
      os.makedirs(outdir)
      args_file = os.path.join(outdir, 'parse_args')
      with open(args_file, 'w') as f: json.dump(args.__dict__, f, indent=2)
      param_file = os.path.join(outdir, 'random_param')
      with open(param_file, 'w') as f: json.dump(random_param, f, indent=2)
      param_file = os.path.join(outdir, 'reward_param')
      with open(param_file, 'w') as f: json.dump(reward_param, f, indent=2)
      param_file = os.path.join(outdir, 'sensor_param')
      with open(param_file, 'w') as f: json.dump(sensor_param, f, indent=2)
      param_file = os.path.join(outdir, 'dynamic_param')
      with open(param_file, 'w') as f: json.dump(dynamic_param, f, indent=2)

    log_openmode = 'w'
    if args.resume != "":
      outdir = os.path.split(args.resume)[0]
      suffix = os.path.split(outdir)[1]
      log_openmode = 'a'
    # logging.basicConfig(filename=os.path.join(outdir, 'log.log'), level=logging.INFO, filemode=log_openmode,
    #                     format='[%(asctime)s %(threadName)s @%(filename)s:%(lineno)d] %(message)s')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter('%(log_color)s[%(asctime)s %(threadName)s @%(filename)s:%(lineno)d] %(message)s',
                                      datefmt='%H:%M:%S'))
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    th = logging.FileHandler(filename=os.path.join(outdir, 'log.log'), mode=log_openmode)
    th.setFormatter(logging.Formatter('[%(asctime)s %(threadName)s @%(filename)s:%(lineno)d] %(message)s',
                                      datefmt='%m-%d %H:%M:%S'))
    th.setLevel(logging.DEBUG)
    logger.addHandler(th)
    logger.info("---------------------------------------------------------------------------------")
    logger.warning("log path: " + outdir)
    logger.info("*********************************************************************************")
    logger.info("argv:{}".format(sys.argv))
    logger.info("args:{}".format(args))
    logger.warning("obs_dim: {}, act_dim: {}".format(obs_dim, action_dim))
    logger.info("reward param: {}".format(reward_param))
    logger.info("dynamic param: {}".format(dynamic_param))
    logger.info("sensor mode: {}".format(sensor_param))

    config = dict(
      outdir=outdir,
      obs_dim=obs_dim,
      etg_path=ETG_path,
      batch_size=BATCH_SIZE,
      actor_lr=ACTOR_LR,
      critic_lr=CRITIC_LR,
      args=args,
      reward_param=reward_param,
      sensor_param=sensor_param
    )
    log_mode = "online" if args.suffix != 'debug' else "disabled"
    # log_mode = "disabled"
    wandb.init(project="ETG_RL", entity="zhenz1996", config=config, save_code=True, name=suffix, mode=log_mode)

    # init w,b(linear layer params)
    ETG_best_param = ES_solver.get_best_param()
    points_add = ETG_best_param.copy().reshape(-1, 2)
    new_points = prior_points + points_add
    w, b, _ = Opt_with_points(ETG=ETG_agent, ETG_T=args.ETG_T, w0=w0, b0=b0, points=new_points)
    plot_gait(w0, b0, ETG_agent, new_points, save_path=os.path.join(outdir, 'init_etg'),
              show_fig=True if args.suffix == 'debug' else False)

    total_steps = 0
    test_flag = 0
    ES_test_flag = 0
    ES_steps = 0
    e_step = args.e_step
    if args.resume != "":
      total_steps = int(os.path.split(args.resume)[1][4:])
      test_flag = (total_steps + 1) // EVAL_EVERY_STEPS
      ES_test_flag = (total_steps + 1) // ES_EVERY_STEPS
    while total_steps < args.max_steps:
      # Train episode
      episode_reward, episode_step, info = run_train_episode(RL_agent, multi_env, rpm, e_step, act_bound, w, b)
      total_steps += episode_step
      wandb.log({'train/episode_reward': episode_reward}, step=total_steps)
      wandb.log({'train/episode_step': episode_step}, step=total_steps)
      for key in info.keys():
        wandb.log({'train/episode_{}'.format(key): info[key]}, step=total_steps)
        if key in reward_param.keys() and reward_param[key] != 0:
          wandb.log({'train/mean_{}'.format(key): info[key] / reward_param[key] / episode_step}, step=total_steps)
        # elif key in Info_Dict.keys():
        #   wandb.log()
      logger.info('[Training] Total Steps: {} Reward: {:.3f} Steps: {} '.format(
        total_steps, episode_reward, episode_step))

      # Evaluate episode, EVAL_EVERY_STEPS=10000
      if (total_steps + 1) // EVAL_EVERY_STEPS >= test_flag:
        while (total_steps + 1) // EVAL_EVERY_STEPS >= test_flag:
          test_flag += 1
          avg_reward, avg_step, info = run_evaluate_episodes(RL_agent, env, 600, act_bound, w, b)
          wandb.log({'eval/episode_reward': avg_reward}, step=total_steps)
          wandb.log({'eval/episode_step': avg_step}, step=total_steps)
          for key in info.keys():
            wandb.log({'eval/episode_{}'.format(key): info[key]}, step=total_steps)
            if key in reward_param.keys() and reward_param[key] != 0:
              wandb.log({'eval/mean_{}'.format(key): info[key] / reward_param[key] / avg_step}, step=total_steps)
          logger.warning('[Evaluation] Over: {} step, Reward: {:.3f} Steps: {}'.format(
            total_steps, avg_reward, avg_step))
        if e_step < 600:
          e_step += 50
        # save RL_agent(SAC) params and ETG(w,b) params
        path = os.path.join(outdir, 'itr_{:d}.pt'.format(int(total_steps)))
        RL_agent.save(path)
        path = os.path.join(outdir, 'itr_{:d}.npz'.format(int(total_steps)))
        np.savez(path, w=w, b=b, param=ETG_best_param)
        path = os.path.join(outdir, 'rpm.npz')
        rpm.save(path)

      # ES episode, ES_EVERY_STEPS=50000, WARMUP_STEPS=10000
      if args.ES and args.ETG and (total_steps + 1) // ES_EVERY_STEPS >= ES_test_flag and total_steps >= WARMUP_STEPS:
        while (total_steps + 1) // ES_EVERY_STEPS >= ES_test_flag:
          ES_test_flag += 1
          best_reward, avg_step, info = run_EStrain_episode(RL_agent, env, 400, act_bound, w, b, rpm)
          logger.info('[ESSteps] Init Reward: {:.3f} step: {}'.format(best_reward, avg_step))
          best_param = ETG_best_param.copy().reshape(-1)
          for ei in range(ES_TRAIN_STEPS):  # ES_TRAIN_STEPS=10
            ES_steps += 1
            fitness_list = []
            steps = []
            infos = {}
            for key in Param_Dict.keys():
              infos[key] = 0
            solutions = ES_solver.ask()  # size: args.popsize(default 40)
            # def _func(solution, env, RL_agent):
            for i, solution in zip(range(len(solutions)), solutions):
              points_add = solution.reshape(-1, 2)
              new_points = prior_points + points_add
              w, b, _ = Opt_with_points(ETG=ETG_agent, ETG_T=args.ETG_T, w0=w0, b0=b0, points=new_points)
              if args.suffix == 'debug':
                plot_gait(w, b, ETG_agent, new_points)
              episode_reward, episode_step, info = run_EStrain_episode(RL_agent, env, 400, act_bound, w, b, rpm)
              fitness_list.append(episode_reward)
              steps.append(episode_step)
              # plot_gait(w, b, ETG_agent, new_points,
              #           save_path=os.path.join(outdir, 'ESSteps{:d}_{:d}_{:.2f}'.format(ES_steps, i, episode_reward)),
              #           show_fig=False)
              for key in infos.keys():
                infos[key] += info[key] / args.popsize
            # reward table
            fitness_list = np.asarray(fitness_list).reshape(-1)
            ES_solver.tell(fitness_list)
            max_index = np.argmax(fitness_list)
            if fitness_list[max_index] > best_reward:
              best_param = solutions[max_index]
              best_reward = fitness_list[max_index]
            results = ES_solver.result()
            sigma = np.mean(results[3])
            logger.info('[ESSteps: {}] Best Reward: {:.3f} step: {} sigma:{:.5f}'.format(
              ES_steps, fitness_list[max_index], steps[max_index], sigma))
            wandb.log({'ES/sigma': sigma})
            wandb.log({'ES/episode_reward': np.mean(fitness_list)})
            wandb.log({'ES/episode_minre': np.min(fitness_list)})
            wandb.log({'ES/episode_maxre': np.max(fitness_list)})
            wandb.log({'ES/episode_restd': np.std(fitness_list)})
            wandb.log({'ES/episode_length': np.mean(steps)})
            for key in Param_Dict.keys():
              wandb.log({'ES/episode_{}'.format(key): infos[key]})
              wandb.log({'ES/mean_{}'.format(key): infos[key] / np.mean(steps)})
        ETG_best_param = best_param
        points_add = ETG_best_param.reshape(-1, 2)
        new_points = prior_points + points_add
        w, b, _ = Opt_with_points(ETG=ETG_agent, ETG_T=args.ETG_T, w0=w0, b0=b0, points=new_points)
        plot_gait(w, b, ETG_agent, new_points,
                  save_path=os.path.join(outdir, 'itr_{:d}'.format(total_steps)), show_fig=False)
        ES_solver.reset(ETG_best_param)
  elif args.eval == 1:
    import rospy
    from sensor_msgs.msg import JointState
    rospy.init_node("etg_eval_node")
    pub_joint = rospy.Publisher("joint_states", JointState, queue_size=100)
    load_dir = args.load[:-3]
    ETG_info = np.load(load_dir + ".npz")
    RL_agent.restore(load_dir + ".pt")
    w = ETG_info["w"]
    b = ETG_info["b"]
    prior_points = ETG_info["param"].reshape(-1, 2)
    # w = w0
    # b = b0
    if not os.path.exists(load_dir):
      os.makedirs(load_dir)
    plot_gait(w, b, ETG_agent, prior_points,
              save_path=os.path.join(load_dir, 'eval_etg'))
    avg_reward, avg_step, info = run_evaluate_episodes(RL_agent, env, 600, act_bound, w, b, pub_joint)
    print("\033[1;32m[Evaluation] Reward: {} Steps: {}\033[0m".format(avg_reward, avg_step))
    print("total reward: ", info)
    origin_rew = {}
    for key in info.keys():
      if Param_Dict[key] != 0.0:
        origin_rew[key] = info[key] / reward_param[key]
      else:
        origin_rew[key] = 0.0
    print("origin reward: ", origin_rew)
    with open(os.path.join(load_dir, 'origin_reward'), 'w') as f:
      json.dump(origin_rew, f, indent=2)
    # record a video for debuging
    outdir = os.path.join(load_dir, args.task_mode)
    os.system("ffmpeg -r 100 -i img/img%01d.jpg -vcodec mpeg4 -vb 40M -y {}.mp4".format(outdir))
    os.system("rm -rf img/*")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--resume", type=str, default="")
  parser.add_argument("--load", type=str, default="", help="Directory to load agent from.")
  parser.add_argument("--eval", type=int, default=0, help="Evaluate or not")
  parser.add_argument("--render", type=int, default=0, help="render or not")
  parser.add_argument("--outdir", type=str, default="train_log")
  parser.add_argument("--suffix", type=str, default="exp0")
  parser.add_argument("--task_mode", type=str, default="heightfield")
  parser.add_argument("--max_steps", type=int, default=3e6)
  parser.add_argument("--env_nums", type=int, default=16)
  parser.add_argument("--learn", type=int, default=8)
  parser.add_argument("--epsilon", type=float, default=0.4)
  parser.add_argument("--gamma", type=float, default=0.95)
  parser.add_argument("--sigma", type=float, default=0.02)
  parser.add_argument("--sigma_decay", type=float, default=0.99)
  parser.add_argument("--popsize", type=float, default=40)
  parser.add_argument("--random", type=int, default=0)
  parser.add_argument("--normal", type=int, default=1)
  parser.add_argument("--footheight", type=float, default=0.08)
  parser.add_argument("--steplen", type=float, default=0.05)
  parser.add_argument("--act_mode", type=str, default="traj")
  parser.add_argument("--act_bound", type=float, default=0.3)
  parser.add_argument("--x_noise", type=int, default=0)
  parser.add_argument("--hft", type=str, default="slope")
  parser.add_argument("--ETG", type=int, default=1)
  parser.add_argument("--ETG_T", type=float, default=0.4)
  parser.add_argument("--ETG_H", type=int, default=20)
  parser.add_argument("--ETG_T2", type=float, default=0.5)
  parser.add_argument("--ETG_path", type=str, default="None")
  parser.add_argument("--vel_d", type=float, default=0.5)
  parser.add_argument("--foot_dx", type=float, default=0.2)
  parser.add_argument("--step_y", type=float, default=0.05)
  parser.add_argument("--reward_p", type=float, default=5)
  parser.add_argument("--enable_action_filter", type=int, default=0)
  parser.add_argument("--action_repeat_num", type=int, default=5)
  parser.add_argument("--ES", type=int, default=1)
  parser.add_argument("--es_rpm", type=int, default=1, help='ES training store into RPM for SAC')
  parser.add_argument("--e_step", type=int, default=400)
  parser.add_argument("--stand", type=float, default=0)
  parser.add_argument("--random_dynamic", type=int, default=0)
  parser.add_argument("--random_force", type=int, default=0)
  parser.add_argument("--rew_torso", type=float, default=1.5)
  parser.add_argument("--rew_up", type=float, default=0.6)
  parser.add_argument("--rew_tau", type=float, default=0.07)
  parser.add_argument("--rew_feet_vel", type=float, default=0.3)
  parser.add_argument("--rew_badfoot", type=float, default=0.1)
  parser.add_argument("--rew_footcontact", type=float, default=0.1)
  parser.add_argument("--sensor_dis", type=int, default=0)
  parser.add_argument("--sensor_motor", type=int, default=1)
  parser.add_argument("--sensor_imu", type=int, default=1)
  parser.add_argument("--sensor_contact", type=int, default=1)
  parser.add_argument("--sensor_ETG", type=int, default=1)
  parser.add_argument("--sensor_ETG_obs", type=int, default=0)
  parser.add_argument("--sensor_footpose", type=int, default=1)
  parser.add_argument("--sensor_dynamic", type=int, default=0)
  parser.add_argument("--sensor_exforce", type=int, default=0)
  parser.add_argument("--sensor_noise", type=int, default=0)
  parser.add_argument("--timesteps", type=int, default=5)
  parser.add_argument("--timeinterval", type=int, default=1)
  parser.add_argument("--RNN_mode", type=str, default="None")
  args = parser.parse_args()
  # resume args from stored files
  if args.resume != "":
    resume_path = args.resume
    load_path = os.path.split(args.resume)[0]
    args_file = os.path.join(load_path, 'parse_args')
    with open(args_file, 'r') as f:
      args.__dict__ = json.load(f)
    args.eval = False
    args.resume = resume_path
    # reload the reward param
    param_file = os.path.join(load_path, 'reward_param')
    with open(param_file, 'r') as f:
      reward_param = json.load(f)
  elif args.eval != 0:
    load_args = args.load
    task_mode = args.task_mode
    hft = args.hft
    load_path = os.path.split(args.load)[0]
    args_file = os.path.join(load_path, 'parse_args')
    with open(args_file, 'r') as f:
      args.__dict__ = json.load(f)
    args.load = load_args
    args.task_mode = task_mode
    args.hft = hft
    args.eval = True
    args.render = True
    param_file = os.path.join(load_path, 'reward_param')
    with open(param_file, 'r') as f:
      reward_param = json.load(f)
  # set params
  random_param['random_dynamics'] = args.random_dynamic
  random_param['random_force'] = args.random_force
  # reward_param['rew_torso'] = args.torso
  # reward_param['rew_up'] = args.up
  # reward_param['rew_tau'] = args.tau
  # reward_param['rew_feet_vel'] = args.feet
  # reward_param['rew_stand'] = args.stand
  # reward_param['rew_badfoot'] = args.badfoot
  # reward_param['rew_footcontact'] = args.footcontact
  sensor_param['dis'] = args.sensor_dis
  sensor_param['motor'] = args.sensor_motor
  sensor_param["imu"] = args.sensor_imu
  sensor_param["contact"] = args.sensor_contact
  sensor_param["ETG"] = args.sensor_ETG
  sensor_param["ETG_obs"] = args.sensor_ETG_obs
  sensor_param["footpose"] = args.sensor_footpose
  sensor_param["dynamic_vec"] = args.sensor_dynamic
  sensor_param["force_vec"] = args.sensor_exforce
  sensor_param["noise"] = args.sensor_noise
  rnn_config = {}
  rnn_config["time_steps"] = args.timesteps
  rnn_config["time_interval"] = args.timeinterval
  rnn_config["mode"] = args.RNN_mode
  sensor_param["RNN"] = rnn_config

  main()
