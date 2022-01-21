import argparse
import json
import logging
import os
import threading
import time
from copy import copy

# from concurrent.futures import ThreadPoolExecutor
# from queue import Queue
import numpy as np
import rospy
from matplotlib import pyplot as plt
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from alg.sac import SAC
from model.mujoco_agent import MujocoAgent
from model.mujoco_model import MujocoModel
from rlschool.quadrupedal.envs.aliengo_robot_env import AliengoRealEnv
from rlschool.quadrupedal.envs.aliengo_robot_env import SENSOR_MODE
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import Param_Dict, Random_Param_Dict
from rlschool.quadrupedal.envs.utilities.ETG_model import ETG_layer
from rlschool.quadrupedal.robots import robot_config


GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ENV_NUMS = 8

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


def Opt_with_points(ETG, ETG_T=0.8, ETG_H=20, points=None, b0=None, w0=None, precision=1e-4, lamb=0.5, **kwargs):
  # ts = [0.5 * ETG_T + 0.1, 0, 0.05, 0.1, 0.15, 0.2]
  ts = [0.5 * ETG_T + 0.2, 0, 0.1, 0.2, 0.3, 0.4]
  if points is None:
    Steplength = kwargs["Steplength"] if "Steplength" in kwargs else 0.05
    Footheight = kwargs["Footheight"] if "Footheight" in kwargs else 0.08
    Penetration = kwargs["Penetration"] if "Penetration" in kwargs else 0.01
    # [[0.0, -0.01], [-0.05, -0.005], [-0.075, 0.06], [0.0, 0.1], [0.075, 0.06], [0.05, -0.005]]
    points = np.array([[0, -Penetration],
                       [-Steplength, -Penetration * 0.5], [-Steplength * 1.0, 0.6 * Footheight],
                       [0, Footheight],
                       [Steplength * 1.0, 0.6 * Footheight], [Steplength, -Penetration * 0.5]])
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


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, max_step, action_bound, w=None, b=None, pub_msgs=False):
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
  # plt.ylim(-0.3, -0.1)
  plt.xlim(0.05, 0.3)

  pub_joint = rospy.Publisher("joint_states", JointState, queue_size=100)
  joint_msg = JointState()
  joint_msg.header.frame_id = "robot"
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

  thread = threading.Thread(target=lambda: rospy.spin())
  thread.start()
  rate = rospy.Rate(100)
  while not rospy.is_shutdown():
  # while not done:
    steps += 1
    t0 = time.time()
    # action = agent.predict(obs)  # NN output: residual control signal
    t1 = time.time()
    pred_time = t1 - t0
    # print("pred time:", pred_time)
    # new_action = copy(action)
    new_action = np.zeros(12)
    obs, reward, done, info = env.step(new_action * action_bound)

    t2 = time.time()
    step_time = t2 - t1
    step_time_all += step_time
    # avg_reward += reward
    # print("step time:", step_time)

    joint_msg.position.clear()
    for i in range(0, 12):
      joint_msg.position.append(info['real_action'][i])  # target joint pos (action)
    for i in range(0, 12):
      # joint_msg.position.append((info['ETG_act'] + info['init_act'])[i])  #
      joint_msg.position.append(info['joint_angle'][i])  # current joint pos
    joint_msg.header.stamp = rospy.Time.now()
    pub_joint.publish(joint_msg)

    # plt.ylim(0.0, 0.1)
    # if info["real_contact"][0] == True:
    #   plt.scatter(info["foot_position_world"][0][0], info["foot_position_world"][0][2], s=5, c='red')
    # else:
    #   plt.scatter(info["foot_position_world"][0][0], info["foot_position_world"][0][2], s=5, c='blue')
    # plot foot trajectory
    # plt.scatter(info["ETG_trj"][0], info["ETG_trj"][2], s=5, c='red')
    # plt.scatter(info["foot_position"][0][0], info["foot_position"][0][2], s=5, c='blue')
    # plt.pause(0.0001)
    # for key in Param_Dict.keys():
    #   if key in info.keys():
    #     if key not in infos.keys():
    #       infos[key] = info[key]
    #     else:
    #       infos[key] += info[key]
    # if steps > max_step:
    #   break
    rate.sleep()
  steps_all += steps
  plt.ioff()
  plt.close(1)
  # print("\033[1;32m[Evaluation] Average step time: {} step/second.\033[0m".format(steps_all/step_time_all))
  logger.debug("[Evaluation] Average step time: {} step/second.".format(steps_all/step_time_all))
  return avg_reward, steps_all, infos


def main():
  rospy.init_node("eval_real_node")

  ## ETG init
  phase = np.array([-np.pi / 2, 0])
  dt = args.action_repeat_num * 0.002  # default: 13 * 0.002 = 0.026s
  args.ETG_T, args.ETG_H = 0.8, 20
  ETG_agent = ETG_layer(args.ETG_T, dt, args.ETG_H, 0.04, phase, 0.2, args.ETG_T2)
  # prior_points: 6x2 [[0.0, -0.01], [-0.05, -0.005], [-0.075, 0.06], [0.0, 0.1], [0.075, 0.06], [0.05, -0.005]]
  w0, b0, prior_points = Opt_with_points(ETG=ETG_agent, ETG_T=args.ETG_T, ETG_H=args.ETG_H,
                                         Footheight=args.footheight, Steplength=args.steplen)
  # plot_gait(w0, b0, ETG_agent, prior_points)
  # if not os.path.exists(ETG_path):
  #   np.savez(ETG_path, w=w0, b=b0, param=prior_points)

  ## create gym envs of the quadruped robot
  env = AliengoRealEnv(motor_control_mode=mode_map[args.act_mode],sensor_mode=sensor_param,
                       normal=args.normal, reward_param=reward_param,
                       ETG=args.ETG, ETG_T=args.ETG_T, ETG_H=args.ETG_H,
                       vel_d=args.vel_d, foot_dx=args.foot_dx, step_y=args.step_y, reward_p=args.reward_p,
                       action_repeat=args.action_repeat_num)
  obs_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

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
  if args.load != "":
    RL_agent.restore(args.load)

  load_dir = args.load[:-3]
  # ETG_info = np.load(load_dir + ".npz")
  # w = ETG_info["w"]
  # b = ETG_info["b"]
  # prior_points = ETG_info["param"].reshape(-1, 2)
  w = w0
  b = b0
  if not os.path.exists(load_dir):
    os.makedirs(load_dir)
  # plot_gait(w, b, ETG_agent, prior_points,
  #           save_path=os.path.join(load_dir, 'eval_etg'))
  avg_reward, avg_step, info = run_evaluate_episodes(RL_agent, env, 600, act_bound, w, b)
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--resume", type=str, default="")
  parser.add_argument("--load", type=str, default="", help="Directory to load agent from.")
  parser.add_argument("--eval", type=int, default=1, help="Evaluate or not")
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
  parser.add_argument("--footheight", type=float, default=0.06)
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
  if args.eval != 0:
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
