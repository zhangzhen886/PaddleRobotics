# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from copy import copy
from rlschool.quadrupedal.envs import locomotion_gym_env
from rlschool.quadrupedal.envs import locomotion_gym_config
from rlschool.quadrupedal.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from rlschool.quadrupedal.envs.env_wrappers import trajectory_generator_wrapper_env
from rlschool.quadrupedal.envs.env_wrappers import simple_openloop
from rlschool.quadrupedal.envs.env_wrappers import simple_forward_task
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import EnvWrapper, Param_Dict, Random_Param_Dict
from rlschool.quadrupedal.envs.env_wrappers.gait_generator_env import GaitGeneratorWrapperEnv
from rlschool.quadrupedal.envs.sensors import robot_sensors
from rlschool.quadrupedal.robots import robot_config, a1, aliengo


SENSOR_MODE = {
  "basepos": 1,  # 3
  "dis": 0,  # 3
  "motor": 1,  # 12+12=24
  "imu": 1,  # 6
  "contact": 1,  # 4
  "footpose": 1,  # 4*3=12
  "ETG": 1,  # 12
  "ETG_obs": 0,  # 20
}

# call in A1GymEnv.init()
def build_regular_env(robot_class,  # a1.A1
                      motor_control_mode,
                      dynamic_param,
                      sensor_mode = {"dis":1,"imu":1,"motor":1,"contact":1},
                      gait = 0,
                      normal = 0,  # Normalisation, (x-mean)/std
                      enable_rendering=False,
                      task_mode = "plane",
                      hf_terrain_mode = "slope",
                      on_rack = False,
                      filter = 0,
                      action_space = 0,
                      random = False,
                      action_limit = (0.75, 0.75, 0.75),
                      wrap_trajectory_generator=True,
                      action_repeat = 13):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.sim_time_step_s = 1. / 500.
  sim_params.num_action_repeat = action_repeat
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  if filter:
    sim_params.enable_action_filter = True
  else:
    sim_params.enable_action_filter = False
  sim_params.enable_action_interpolation = False
  sim_params.enable_clip_motor_commands = False
  sim_params.robot_on_rack = on_rack
  gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)

  # choice sensors(observation variables), called in
  sensors = []
  noise = True if ("noise" in sensor_mode and sensor_mode["noise"]) else False
  dt = sim_params.num_action_repeat * sim_params.sim_time_step_s  # 13 * 0.002 = 0.026
  if sensor_mode["dis"]:  # 3
    sensors.append(robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True,normal=normal,noise=noise,dt=dt))
  if sensor_mode["imu"]==1:  # 6
    sensors.append(robot_sensors.IMUSensor(channels=["R", "P", "Y","dR", "dP", "dY"],normal=normal,noise=noise))
  elif sensor_mode["imu"]==2:  # 3
    sensors.append(robot_sensors.IMUSensor(channels=["dR", "dP", "dY"],noise=noise))
  if sensor_mode["motor"]==1:  # 12+12=24
    sensors.append(robot_sensors.MotorAngleAccSensor(num_motors=aliengo.NUM_MOTORS,normal=normal,noise=noise,dt=dt))
  elif sensor_mode["motor"]==2:  # 12
    sensors.append(robot_sensors.MotorAngleSensor(num_motors=aliengo.NUM_MOTORS,noise=noise))
  if sensor_mode["contact"] == 1:  # 4
    sensors.append(robot_sensors.FootContactSensor())
  elif sensor_mode["contact"] == 2:  # 8
    sensors.append(robot_sensors.SimpleFootForceSensor())
  if sensor_mode["footpose"]:  # 4*3=12
    sensors.append(robot_sensors.FootPoseSensor(normal=False))
  if "basepos"in sensor_mode and sensor_mode["basepos"]:  # 3
    sensors.append(robot_sensors.BasePositionSensor())

  task = simple_forward_task.SimpleForwardTask(dynamic_param)
  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            param = dynamic_param,
                                            robot_class=robot_class,
                                            robot_sensors=sensors,
                                            random=random,
                                            task=task,
                                            task_mode=task_mode,
                                            hf_terrain_mode=hf_terrain_mode)
  env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  if gait != 0 and (motor_control_mode
                    == robot_config.MotorControlMode.POSITION):
    env = GaitGeneratorWrapperEnv(env, gait_mode=gait)
  elif (motor_control_mode
        == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
      env,
      trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
        action_limit=0.75, action_space=action_space))  # origin action_limit=action_limit

  return env


class A1GymEnv(gym.Env):
  """A1 environment that supports the gym interface."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self,
               task="plane",
               hf_terrain_mode="slope",
               motor_control_mode=robot_config.MotorControlMode.POSITION,
               render=False,
               on_rack=False,
               sensor_mode=SENSOR_MODE,
               gait=0,
               normal=0,
               filter_=0,
               action_space=0,
               random_dynamic=False,
               reward_param=Param_Dict,
               random_param=Random_Param_Dict,
               dynamic_param={},
               ETG=0,
               ETG_T=0.5,
               ETG_H=20,
               ETG_path="",
               vel_d=0.6,
               foot_dx=0.2,
               step_y=0.05,
               reward_p=1.0,
               action_limit=(0.75, 0.75, 0.75),
               action_repeat=13,
               **kwargs):
    self._env = build_regular_env(
      # a1.A1,
      aliengo.Aliengo,
      motor_control_mode=motor_control_mode,
      gait=gait,
      normal=normal,
      task_mode=task,
      hf_terrain_mode=hf_terrain_mode,
      enable_rendering=render,
      action_limit=action_limit,
      sensor_mode=sensor_mode,
      random=random_dynamic,
      filter=filter_,
      action_space=action_space,
      on_rack=on_rack,
      dynamic_param=dynamic_param,
      action_repeat=action_repeat,
    )
    self._env = EnvWrapper(
      env=self._env,
      reward_param=reward_param,
      sensor_mode=sensor_mode,
      normal=normal,
      ETG_T=ETG_T,
      ETG_H=ETG_H,
      ETG_path=ETG_path,
      ETG=ETG,
      random_param=random_param,
      vel_d=vel_d,
      foot_dx=foot_dx,
      step_y=step_y,
      task_mode=task,
      reward_p=reward_p,
    )
    self.observation_space = self._env.observation_space
    self.share_observation_space = copy(self.observation_space)
    self.action_space = self._env.action_space
    self.sensor_mode = sensor_mode

  def step(self, action):
    return self._env.step(action)

  def step(self, action, **kwargs):
    return self._env.step(action, **kwargs)

  def reset(self, **kwargs):
    return self._env.reset(**kwargs)

  def close(self):
    self._env.close()

  def render(self, mode):
    return self._env.render(mode)

  def __getattr__(self, attr):
    return getattr(self._env, attr)
