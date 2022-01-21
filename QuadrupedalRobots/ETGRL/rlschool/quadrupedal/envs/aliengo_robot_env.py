# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
import collections
import threading
import time

import gym
from gym import spaces
import pybullet as p
import numpy as np
from copy import copy

import rospy
from std_msgs.msg import Float32MultiArray
from unitree_legged_msgs.msg import LowState

from rlschool.quadrupedal.robots import a1
from rlschool.quadrupedal.robots import aliengo
from rlschool.quadrupedal.robots import robot_config
from rlschool.quadrupedal.robots import action_filter
from rlschool.quadrupedal.envs.sensors import sensor
from rlschool.quadrupedal.envs.sensors import robot_sensors
from rlschool.quadrupedal.envs.sensors import space_utils
from rlschool.quadrupedal.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from rlschool.quadrupedal.envs.env_wrappers import trajectory_generator_wrapper_env
from rlschool.quadrupedal.envs.env_wrappers import simple_openloop
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import ETGWrapper, ObservationWrapper, RewardShaping
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import Param_Dict
from rlschool.quadrupedal.envs import locomotion_gym_config


SENSOR_MODE = {
  "dis": 0,  # 3
  "motor": 1,  # 12+12=24
  "imu": 1,  # 6
  "contact": 1,  # 4
  "footpose": 1,  # 4*3=12
  "basepos": 1,  # 3
  "ETG": 1,  # 12
  "ETG_obs": 0,  # 20
}

NUM_MOTORS = 12
NUM_LEGS = 4
DOFS_PER_LEG = 3
INIT_MOTOR_ANGLES = np.array([0, 0.77, -1.59] * NUM_LEGS)
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
DEFAULT_HIP_POSITIONS = (
  (0.2399, -0.051, 0),
  (0.2399, 0.051, 0),
  (-0.2399, -0.051, 0),
  (-0.2399, 0.051, 0),
)
BASE_FOOT = np.array(
  [0.2399, -0.134, -0.35,
   0.2399, 0.134, -0.35,
   -0.2399, -0.134, -0.35,
   -0.2399, 0.134, -0.35],
)

COM_OFFSET = -np.array([0.0, 0.0, 0.0])
HIP_OFFSETS = np.array(DEFAULT_HIP_POSITIONS) + COM_OFFSET

LENGTH_HIP_LINK = 0.083
LENGTH_UPPER_LINK = 0.25
LENGTH_LOWER_LINK = 0.25

ABDUCTION_P_GAIN = 40.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 50.0
HIP_D_GAIN = 1.0
KNEE_P_GAIN = 50.0
KNEE_D_GAIN = 1.0

def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
  l_up = LENGTH_UPPER_LINK
  l_low = LENGTH_LOWER_LINK
  l_hip = LENGTH_HIP_LINK * l_hip_sign
  x, y, z = foot_position[0], foot_position[1], foot_position[2]
  x += 0.008
  theta_knee = -np.arccos(
      (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
      (2 * l_low * l_up))
  l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
  theta_hip = np.arcsin(-x / l) - theta_knee / 2
  c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
  s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = np.arctan2(s1, c1)
  return np.array([theta_ab, theta_hip, theta_knee])


def foot_position_in_hip_frame(angles, l_hip_sign=1):
  theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
  l_up = LENGTH_UPPER_LINK
  l_low = LENGTH_LOWER_LINK
  l_hip = LENGTH_HIP_LINK * l_hip_sign
  leg_distance = np.sqrt(l_up**2 + l_low**2 +
                         2 * l_up * l_low * np.cos(theta_knee))
  eff_swing = theta_hip + theta_knee / 2

  off_x_hip = -leg_distance * np.sin(eff_swing)
  off_z_hip = -leg_distance * np.cos(eff_swing)
  off_y_hip = l_hip

  off_x = off_x_hip
  off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
  off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
  return np.array([off_x, off_y, off_z])


def foot_positions_in_base_frame(foot_angles):
  foot_angles = foot_angles.reshape((4, 3))
  foot_positions = np.zeros((4, 3))
  for i in range(4):
    foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                   l_hip_sign=(-1)**(i + 1))
  return foot_positions + HIP_OFFSETS


class RealRobot(gym.Env):
  ACTION_CONFIG = [
    locomotion_gym_config.ScalarField(name="FR_hip_motor",
                                      upper_bound=0.802851455917,
                                      lower_bound=-0.802851455917),
    locomotion_gym_config.ScalarField(name="FR_upper_joint",
                                      upper_bound=4.18879020479,
                                      lower_bound=-1.0471975512),
    locomotion_gym_config.ScalarField(name="FR_lower_joint",
                                      upper_bound=-0.916297857297,
                                      lower_bound=-2.69653369433),
    locomotion_gym_config.ScalarField(name="FL_hip_motor",
                                      upper_bound=0.802851455917,
                                      lower_bound=-0.802851455917),
    locomotion_gym_config.ScalarField(name="FL_upper_joint",
                                      upper_bound=4.18879020479,
                                      lower_bound=-1.0471975512),
    locomotion_gym_config.ScalarField(name="FL_lower_joint",
                                      upper_bound=-0.916297857297,
                                      lower_bound=-2.69653369433),
    locomotion_gym_config.ScalarField(name="RR_hip_motor",
                                      upper_bound=0.802851455917,
                                      lower_bound=-0.802851455917),
    locomotion_gym_config.ScalarField(name="RR_upper_joint",
                                      upper_bound=4.18879020479,
                                      lower_bound=-1.0471975512),
    locomotion_gym_config.ScalarField(name="RR_lower_joint",
                                      upper_bound=-0.916297857297,
                                      lower_bound=-2.69653369433),
    locomotion_gym_config.ScalarField(name="RL_hip_motor",
                                      upper_bound=0.802851455917,
                                      lower_bound=-0.802851455917),
    locomotion_gym_config.ScalarField(name="RL_upper_joint",
                                      upper_bound=4.18879020479,
                                      lower_bound=-1.0471975512),
    locomotion_gym_config.ScalarField(name="RL_lower_joint",
                                      upper_bound=-0.916297857297,
                                      lower_bound=-2.69653369433),
  ]
  def __init__(
      self,
      num_motors=NUM_MOTORS,
      dofs_per_leg=DOFS_PER_LEG,
      time_step=0.002,
      action_repeat=10,
      sensors=None,
      enable_clip_motor_commands=False,
      enable_action_filter=True,
      enable_action_interpolation=True,
      motor_torque_limits=10.0,
      allow_knee_contact=False,
  ):
    self.num_motors = num_motors
    self.num_legs = self.num_motors // dofs_per_leg
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands
    self._action_repeat = action_repeat
    self._time_step = time_step
    self._observed_motor_torques = np.zeros(num_motors)
    self._applied_motor_torques = np.zeros(num_motors)
    self._enable_action_interpolation = enable_action_interpolation
    self._enable_action_filter = enable_action_filter
    self._last_action = None

    self.base_foot_position = BASE_FOOT
    self.motor_kp = [
      ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
      ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
    ]
    self.motor_kd = [
      ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
      ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
    ]
    self._torque_limits = motor_torque_limits
    self._step_counter = 0
    self._state_action_counter = 0
    self._joint_states = np.zeros(36)
    self._base_position = np.zeros(3)
    self._base_velocity = np.zeros(3)
    self._base_orientation = np.zeros(4)
    self._base_gyroscope = np.zeros(3)
    self._foot_force = np.zeros(4)
    self._observation_valid = False

    self.SetAllSensors(sensors if sensors is not None else list())

    if motor_torque_limits is not None:
      if isinstance(motor_torque_limits, (collections.Sequence, np.ndarray)):
        self._torque_limits = np.asarray(motor_torque_limits)
      else:
        self._torque_limits = np.full(num_motors, motor_torque_limits)

    if self._enable_action_filter:
      self._action_filter = self._BuildActionFilter()
      self._ResetActionFilter()

    self.robot_cmd_pub = rospy.Publisher('low_cmd', Float32MultiArray, queue_size=1)
    rospy.Subscriber("low_state", LowState, callback=self.ReceiveObservation, queue_size=1)
    self.msg_time = rospy.Time.now()
    self.threadLock = threading.Lock()

  def GetTimeSinceReset(self):
    # print("time_step:",self.time_step)
    return self._step_counter * self._time_step

  def ReceiveObservation(self, msgs):
    time_now = rospy.Time.now()
    # print("msg duration: ", (time_now-self.msg_time).to_sec()*1000)
    self.msg_time = time_now
    self.threadLock.acquire()
    for i in range(self.num_motors):
      self._joint_states[i] = np.asarray(msgs.motorState[i].q)
      self._joint_states[12+i] = np.asarray(msgs.motorState[i].dq)
      self._joint_states[24+i] = np.asarray(msgs.motorState[i].tauEst)
    self._base_position = np.zeros(3)
    self._base_velocity = np.zeros(3)
    self._base_orientation = np.asarray(msgs.imu.quaternion)
    self._base_gyroscope = np.asarray(msgs.imu.gyroscope)
    self._foot_force = np.asarray(msgs.footForce)
    self.threadLock.release()
    if not self._observation_valid:
      self._observation_valid = True

  def IsObservationValid(self):
    return self._observation_valid

  def GetObservation(self):
    observations = {}
    observations['IMU'] = np.concatenate((self.GetBaseRollPitchYaw(), self.GetBaseRollPitchYawRate()))
    observations['MotorAngleAcc'] = np.concatenate((self.GetMotorAngles(), self.GetMotorVelocities()))
    observations['FootContactSensor'] = self.GetFootContactsForce(mode='simple')
    # observations['FootPoseSensor'] = self.GetFootPositionsInBaseFrame()
    return observations

  def _GetMotorObservation(self):
    self.threadLock.acquire()
    q = self._joint_states[0:12]  # 0~12, MotorAngles
    qdot = self._joint_states[12:24]  # 12~24, MotorVelocities
    self.threadLock.release()
    return (np.array(q), np.array(qdot))

  def GetFootContactsForce(self,mode='simple'):
    simplecontact = np.zeros(8)
    for m in range(4):
      if self._foot_force[m] != 0:
        simplecontact[m] = True if self._foot_force[m] != 0 else False
        simplecontact[m+4] = self._foot_force[m]
    # print('simple',simplecontact)
    return simplecontact

  def GetBasePosition(self):
    return self._base_position

  def GetBaseVelocity(self):
    return self._base_velocity

  def GetBaseOrientation(self):
    return self._base_orientation

  def GetBaseRollPitchYaw(self):
    roll_pitch_yaw = p.getEulerFromQuaternion(self.GetBaseOrientation())
    return np.asarray(roll_pitch_yaw)

  def GetBaseRollPitchYawRate(self):
    return self._base_gyroscope

  def GetFootPositionsInBaseFrame(self):
    return np.zeros(12)

  def GetFootContacts(self):
    return np.array(self._foot_force, dtype=bool)

  def GetMotorAngles(self):
    return np.asarray(self._joint_states[0:12])

  def GetMotorVelocities(self):
    return np.asarray(self._joint_states[12:24])

  def GetMotorTorques(self):
    return np.asarray(self._joint_states[24:36])

  def GetEnergyConsumptionPerControlStep(self):
    return np.abs(np.dot(
      self.GetMotorTorques(),
      self.GetMotorVelocities())) * self._time_step * self._action_repeat

  def GetCostOfTransport(self):
    tv = self.GetMotorTorques() * self.GetMotorVelocities()
    tv[tv < 0] = 0
    return tv.sum() / (np.linalg.norm(self.GetBaseVelocity()) * 20.0 * 9.8)

  def SetAllSensors(self, sensors):
    for s in sensors:
      s.set_robot(self)
    self._sensors = sensors

  def GetAllSensors(self):
    return self._sensors

  def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                              foot_local_position):
    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = foot_position_in_hip_frame_to_joint_angle(
        foot_local_position - HIP_OFFSETS[leg_id],
        l_hip_sign=(-1)**(leg_id + 1))
    # Return the joing index (the same as when calling GetMotorAngles) as well as the angles.
    return joint_position_idxs, joint_angles.tolist()

  def _BuildActionFilter(self):
    sampling_rate = 1 / (self._time_step * self._action_repeat)
    num_joints = self.num_motors
    a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,
                                                num_joints=num_joints)
    return a_filter

  def _ResetActionFilter(self):
    self._action_filter.reset()

  def _FilterAction(self, action):
    if self._step_counter == 0:
      default_action, _ = self._GetMotorObservation()
      self._action_filter.init_history(default_action)
    filtered_action = self._action_filter.filter(action)
    return filtered_action

  def _ApplyAction(self, motor_commands):
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)

    self.last_action_time = self._state_action_counter * self._time_step
    motor_commands = np.asarray(motor_commands)

    motor_angle, motor_velocity = self._GetMotorObservation()  # MotorAngles and MotorVelocities
    desired_motor_angles = motor_commands
    desired_motor_velocities = np.full(12, 0)
    # print("desired motor_angle: {:.3f}, current: {:.3f}".format(desired_motor_angles[1], motor_angle[1]))
    kp = self.motor_kp
    kd = self.motor_kd
    motor_torques = -1 * (kp * (motor_angle - desired_motor_angles)) - kd * (
        motor_velocity - desired_motor_velocities)
    if self._torque_limits is not None:
      if len(self._torque_limits) != len(motor_torques):
        raise ValueError(
            "Torque limits dimension does not match the number of motors.")
      motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
                              self._torque_limits)
    # print('motor_t:',motor_torques)
    return motor_torques

  def _ClipMotorCommands(self, motor_commands):
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles, _ = self._GetMotorObservation()
    motor_commands = np.clip(motor_commands,
                             current_motor_angles - max_angle_change,
                             current_motor_angles + max_angle_change)
    return motor_commands

  def _ProcessAction(self, action, substep_count):
    if self._enable_action_interpolation and self._last_action is not None:
      lerp = float(substep_count + 1) / self._action_repeat
      proc_action = self._last_action + lerp * (action - self._last_action)
    else:
      proc_action = action
    return proc_action

  def _StepInternal(self, action):
    t = self._ApplyAction(action)
    self._state_action_counter += 1
    return t

  def Step(self, action):
    filt_action = action
    if self._enable_action_filter:
      filt_action = self._FilterAction(action)
    torques = []
    cmd_msgs = Float32MultiArray()
    # calculate joint torques with PD controller
    for i in range(self._action_repeat):
      time0 = time.time()
      # print("recv duration: ", (time0-self.msg_time.to_sec())*1000.0)
      proc_action = self._ProcessAction(filt_action, i)
      # print("ori_act: {:.3f}, filt_act: {:.3f}, proc_act: {:.3f}".format(action[1], filt_action[1], proc_action[1]))
      t = self._StepInternal(proc_action)
      torques.append(t)
      cmd_msgs.data = t.tolist()
      self.robot_cmd_pub.publish(cmd_msgs)
      self._step_counter += 1
      time_duration = time.time()-time0
      if time_duration < self._time_step:
        time.sleep(self._time_step-time_duration)

    self._last_action = action
    return torques

  def Terminate(self):
    pass

class LocomotionRealEnv(gym.Env):
  def __init__(
      self,
      time_step=0.002,
      action_repeat=5,
      env_sensors=None,
      robot_sensors=None,
  ):
    self._num_action_repeat = action_repeat
    self._last_action = None
    self._robot_time_step = time_step

    self._robot_sensors = robot_sensors
    self._env_sensors = env_sensors if env_sensors is not None else list()

    self._last_frame_time = 0.0
    self._env_step_counter = 0
    self._env_time_step = self._num_action_repeat * self._robot_time_step

    self.reset()
    # The action list contains the name of all actions.
    self._build_action_space()
    self.observation_space = (
      space_utils.convert_sensors_to_gym_space_dictionary(
        self.all_sensors()))

  def reset(self, **kwargs):
    self._robot = RealRobot(
      sensors=self._robot_sensors,
      time_step=self._robot_time_step,
      action_repeat = self._num_action_repeat,
      enable_clip_motor_commands=False,
      enable_action_filter=False,
      enable_action_interpolation=False,
    )
    for sensor in self._robot_sensors:
      sensor.reset()
    while not self._robot.IsObservationValid():
      rospy.logwarn_throttle(0.5, "waiting observation valid...")
      pass
    info = {}
    info["env_info"] = None
    info["rot_quat"] = self._robot.GetBaseOrientation()
    info["rot_mat"] = p.getMatrixFromQuaternion(info["rot_quat"])
    info["rot_euler"] = self._robot.GetBaseRollPitchYaw()
    info["drpy"] = self._robot.GetBaseRollPitchYawRate()
    info["base_position"] = self._robot.GetBasePosition()
    info["foot_position"] = self._robot.GetFootPositionsInBaseFrame()
    info["real_contact"] = self._robot.GetFootContacts()
    info["joint_angle"] = self._robot.GetMotorAngles()
    info["joint_torque"] = self._robot.GetMotorTorques()
    info["energy"] = self._robot.GetEnergyConsumptionPerControlStep()
    info["transport_cost"] = self._robot.GetCostOfTransport()
    return self._get_observation(), info

  def step(self, action):
    self._last_action = action
    # time_spent = time.time() - self._last_frame_time
    # self._last_frame_time = time.time()
    # print("[step] time_spent: ", time_spent)
    torques = self._robot.Step(action)
    for s in self.all_sensors():
      s.on_step(self)
    done = self._termination()
    self._env_step_counter += 1
    if done:
      self._robot.Terminate()

    info = {}
    info["env_info"] = None
    info["rot_quat"] = self._robot.GetBaseOrientation()
    info["rot_mat"] = p.getMatrixFromQuaternion(info["rot_quat"])
    info["rot_euler"] = self._robot.GetBaseRollPitchYaw()
    info["drpy"] = self._robot.GetBaseRollPitchYawRate()
    info["base_position"] = self._robot.GetBasePosition()
    info["foot_position"] = self._robot.GetFootPositionsInBaseFrame()
    info["real_contact"] = self._robot.GetFootContacts()
    info["joint_angle"] = self._robot.GetMotorAngles()
    info["joint_torque"] = self._robot.GetMotorTorques()
    info["energy"] = self._robot.GetEnergyConsumptionPerControlStep()
    info["transport_cost"] = self._robot.GetCostOfTransport()
    return self._get_observation(), None, done, info

  def all_sensors(self):
    """Returns all robot and environmental sensors."""
    return self._robot.GetAllSensors() + self._env_sensors

  def _build_action_space(self):
    """Builds action space based on motor control mode."""
    # Position mode
    action_upper_bound = []
    action_lower_bound = []
    action_config = self._robot.ACTION_CONFIG
    for action in action_config:
      action_upper_bound.append(action.upper_bound)
      action_lower_bound.append(action.lower_bound)
    self.action_space = spaces.Box(np.array(action_lower_bound),
                                   np.array(action_upper_bound),
                                   dtype=np.float32)

  def _termination(self):
    return False

  def _get_observation(self):
    sensors_dict = {}
    for s in self.all_sensors():
      sensors_dict[s.get_name()] = s.get_observation()
    observations = collections.OrderedDict(sorted(list(sensors_dict.items())))
    # print(observations)
    return observations

  def get_time_since_reset(self):
    return self._robot.GetTimeSinceReset()

  @property
  def robot(self):
    return self._robot

  @property
  def env_time_step(self):
    return self._env_time_step

class AliengoRealEnv(gym.Env):
  """A1 environment that supports the gym interface."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self,
               motor_control_mode=robot_config.MotorControlMode.POSITION,
               sensor_mode=SENSOR_MODE,
               normal=False,
               action_space=0,
               reward_param=Param_Dict,
               ETG=0,
               ETG_T=0.5,
               ETG_H=20,
               ETG_path="",
               vel_d=0.6,
               foot_dx=0.2,
               step_y=0.05,
               reward_p=1.0,
               action_repeat=10,
               **kwargs):
    # choice sensors(observation variables), called in
    sensors = []
    dt = action_repeat * 0.002
    if sensor_mode["imu"] == 1:
      sensors.append(robot_sensors.IMUSensor(channels=["R", "P", "Y", "dR", "dP", "dY"], normal=normal, noise=False))
    if sensor_mode["motor"] == 1:
      sensors.append(robot_sensors.MotorAngleAccSensor(num_motors=aliengo.NUM_MOTORS, normal=normal, noise=False, dt=dt))
    if sensor_mode["contact"] == 1:
      sensors.append(robot_sensors.FootContactSensor())
    if sensor_mode["footpose"]:
      sensors.append(robot_sensors.FootPoseSensor(normal=True))
    if sensor_mode["basepos"]:
      sensors.append(robot_sensors.BasePositionSensor())

    self._env = LocomotionRealEnv(
      action_repeat=action_repeat,
      robot_sensors=sensors,
    )
    self._env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(self._env)
    self._env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
      self._env, trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_space=action_space))

    self._env = ETGWrapper(env=self._env, ETG=ETG, ETG_T=ETG_T, ETG_path=ETG_path,
                     ETG_T2=0.5, ETG_H=ETG_H, step_y=step_y)
    self._env = ObservationWrapper(env=self._env, ETG=ETG, sensor_mode=sensor_mode, normal=normal, ETG_H=ETG_H)
    # self._env = RewardShaping(env=self._env, reward_param=reward_param, reward_p=reward_p, foot_dx=foot_dx)

    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

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

if __name__ == "__main__":
  rospy.init_node("test_robot_env")
  env = LocomotionRealEnv()

  thread = threading.Thread(target=lambda:rospy.spin())
  thread.start()

  rate = rospy.Rate(100)
  while True:
    env.step(action=INIT_MOTOR_ANGLES)
    rate.sleep()


