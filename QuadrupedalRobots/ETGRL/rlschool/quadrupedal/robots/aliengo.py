# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation

"""Pybullet simulation of a Laikago robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import re
import numpy as np
import pybullet as pyb  # pytype: disable=import-error

from rlschool.quadrupedal.robots import laikago_constants
from rlschool.quadrupedal.robots import laikago_motor
from rlschool.quadrupedal.robots import minitaur
from rlschool.quadrupedal.robots import robot_config
from rlschool.quadrupedal.envs import locomotion_gym_config

NUM_MOTORS = 12
NUM_LEGS = 4
DOFS_PER_LEG = 3
MOTOR_NAMES = [
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.37]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
# Bases on the readings from Laikago's default pose.
# INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)
INIT_MOTOR_ANGLES = np.array([0, 0.77, -1.59] * NUM_LEGS)

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
DEFAULT_HIP_POSITIONS = (
  (0.2399, -0.051, 0),
  (0.2399, 0.051, 0),
  (-0.2399, -0.051, 0),
  (-0.2399, 0.051, 0),
)

LENGTH_HIP_LINK = 0.083
LENGTH_UPPER_LINK = 0.25
LENGTH_LOWER_LINK = 0.25

COM_OFFSET = -np.array([0.0, 0.0, 0.0])
HIP_OFFSETS = np.array([[0.2399, -0.051, 0.0], [0.2399, 0.051, 0.0],
                        [-0.2399, -0.051, 0.0], [-0.2399, 0.051, 0.0]
                        ]) + COM_OFFSET
BASE_FOOT = np.array([0.2399, -0.134, -0.35,
                      0.2399, 0.134, -0.35,
                      -0.2399, -0.134, -0.35,
                      -0.2399, 0.134, -0.35])

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 2.
HIP_P_GAIN = 150.0
HIP_D_GAIN = 4.0
KNEE_P_GAIN = 150.0
KNEE_D_GAIN = 4.0

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_thigh_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_calf_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_foot\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = os.environ['HOME']+"/workspaces/PaddleRobotics/QuadrupedalRobots/ETGRL/aliengo/aliengo.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3


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


def analytical_leg_jacobian(leg_angles, leg_id):
  """
  Computes the analytical Jacobian.
  Args:
  ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    l_hip_sign: whether it's a left (1) or right(-1) leg.
  """
  l_up = LENGTH_UPPER_LINK
  l_low = LENGTH_LOWER_LINK
  l_hip = LENGTH_HIP_LINK * (-1)**(leg_id + 1)

  t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
  l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
  t_eff = t2 + t3 / 2
  J = np.zeros((3, 3))
  J[0, 0] = 0
  J[0, 1] = -l_eff * np.cos(t_eff)
  J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
      t_eff) / 2
  J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
  J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
  J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
      t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
  J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
  J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
  J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
      t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
  return J


# For JIT compilation
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), 1)
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), -1)


class Aliengo(minitaur.Minitaur):
  """A simulation for the Aliengo robot."""
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
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=None,
      reset_time=1,
      allow_knee_contact=False,
  ):
    self._urdf_filename = urdf_filename
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands
    self.base_foot_position = BASE_FOOT

    motor_kp = [
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
        HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
    ]
    motor_kd = [
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
        HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
    ]

    super(Aliengo, self).__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_control_mode=motor_control_mode,
        motor_model_class=laikago_motor.LaikagoMotorModel,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        on_rack=on_rack,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter,
        reset_time=reset_time,
    )

  def _LoadRobotURDF(self):
    a1_urdf_path = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          a1_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
        a1_urdf_path,
        self._GetDefaultInitPosition(),
        self._GetDefaultInitOrientation())

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()
    if reset_time <= 0:
      return

    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
          motor_control_mode=robot_config.MotorControlMode.POSITION)

    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self._StepInternal(
            default_motor_angles,
            motor_control_mode=robot_config.MotorControlMode.POSITION)

  def GetHipPositionsInBaseFrame(self):
    return DEFAULT_HIP_POSITIONS

  def GetFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
    contacts = [False, False, False, False]
    for contact in all_contacts:
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(
            contact[_LINK_A_FIELD_NUMBER])
        contacts[toe_link_index] = True
      except ValueError:
        continue
    return contacts
  
  def GetBadFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
    bad_num = 0
    for contact in all_contacts:
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      elif contact[_LINK_A_FIELD_NUMBER] %5 != 0 :
        bad_num += 1
    return bad_num
  
  def GetFootContactsForce(self,mode='simple'):
    # contact state(1), normalforce(3),friction_force(3)
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
    contacts = np.zeros((4,4))
    for contact in all_contacts:
      # print(contact)
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        # print('self!')
        continue
      try:
        toe_link_index = self._foot_link_ids.index(
            contact[_LINK_A_FIELD_NUMBER])
        # contacts[toe_link_index] = True
        contacts[toe_link_index,0]=1
        normalForce = contact[9]*np.asarray(contact[7])
        # frictionForce = contact[10]*np.asarray(contact[11])+contact[12]*np.asarray(contact[13])
        # print('link:',toe_link_index,'normal:',normalForce,'friction:',frictionForce)
        for i in range(3):
          contacts[toe_link_index,i+1]+=normalForce[i]
          # contacts[toe_link_index,i+4]+=frictionForce[i]
        # print('contact:',contacts[toe_link_index])
      except ValueError:
        continue
    # print('contact:',contacts)
    simplecontact = np.zeros(8)
    if mode == 'simple':
        for m in range(4):
          simplecontact[m] = contacts[m,0]
          simplecontact[m+4] = np.linalg.norm(contacts[m,1:])/100.0
        # print('simple',simplecontact)
        return simplecontact
    else:
      return contacts.reshape(-1)

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      if "hip_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
      elif "thigh_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
      elif "calf_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
      else:
        raise ValueError("The name %s is not recognized as a motor joint." %
                         name)
      self._pybullet_client.resetJointState(self.quadruped,
                                            self._joint_name_to_id[name],
                                            angle,
                                            targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

  def _RecordMassInfoFromURDF(self):
    """Records the mass information from the URDF file."""
    self._base_mass_urdf = []
    for chassis_id in self._chassis_link_ids:
      self._base_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
    self._leg_masses_urdf = []
    for hip_id in self._hip_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, hip_id)[0])
    for motor_id in self._thigh_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])
    for lower_id in self._calf_link_ids:
      self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped, lower_id)[0])
    for foot_id in self._foot_link_ids:
      self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped, foot_id)[0])
    self._robot_mass = np.sum(self._base_mass_urdf) + np.sum(self._leg_masses_urdf)

  def _RecordInertiaInfoFromURDF(self):
    """Record the inertia of each body from URDF file."""
    self._link_urdf = []
    num_bodies = self._pybullet_client.getNumJoints(self.quadruped)
    for body_id in range(-1, num_bodies):  # -1 is for the base link.
      inertia = self._pybullet_client.getDynamicsInfo(self.quadruped,
                                                      body_id)[2]
      self._link_urdf.append(inertia)
    # We need to use id+1 to index self._link_urdf because it has the base
    # (index = -1) at the first element.
    self._base_inertia_urdf = [
        self._link_urdf[chassis_id + 1] for chassis_id in self._chassis_link_ids
    ]
    self._leg_inertia_urdf = [
        self._link_urdf[leg_id + 1] for leg_id in self._hip_link_ids
    ]
    self._leg_inertia_urdf.extend(
        [self._link_urdf[leg_id + 1] for leg_id in self._thigh_link_ids]
    )
    self._leg_inertia_urdf.extend(
      [self._link_urdf[leg_id + 1] for leg_id in self._calf_link_ids]
    )
    self._leg_inertia_urdf.extend(
      [self._link_urdf[leg_id + 1] for leg_id in self._foot_link_ids]
    )

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.
       Called in the "Reset()" func
    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)  # 21=4*5+1
    self._hip_link_ids = []  # '_hip_joint, _hip_fixed': 1, 2, 6, 7, 11, 12, 16, 17
    self._thigh_link_ids = []  # '_thigh_joint': 3, 8, 13, 18
    self._calf_link_ids = []  # 'calf_joint': 4, 9, 14, 19
    self._foot_link_ids = []  # '_toe_fixed': 5, 10, 15, 20
    self._motor_link_ids = []  # 1, 2, 6, 7, 11, 12, 16, 17
    self._leg_link_ids = []  # 4, 5, 9, 10, 14, 15, 19, 20
    self._imu_link_ids = []  # 0

    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if HIP_NAME_PATTERN.match(joint_name):
        self._hip_link_ids.append(joint_id)
      elif UPPER_NAME_PATTERN.match(joint_name):
        self._thigh_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif LOWER_NAME_PATTERN.match(joint_name):
        self._calf_link_ids.append(joint_id)
      elif TOE_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      elif IMU_NAME_PATTERN.match(joint_name):
        self._imu_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._motor_link_ids.extend(self._hip_link_ids)
    self._motor_link_ids.extend(self._thigh_link_ids)
    self._leg_link_ids.extend(self._calf_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)

    #assert len(self._foot_link_ids) == NUM_LEGS
    self._hip_link_ids.sort()
    self._thigh_link_ids.sort()
    self._calf_link_ids.sort()
    self._foot_link_ids.sort()
    self._motor_link_ids.sort()
    self._leg_link_ids.sort()

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    else:
      return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    # The Laikago URDF assumes the initial pose of heading towards z axis,
    # and belly towards y axis. The following transformation is to transform
    # the Laikago initial orientation to our commonly used orientation: heading
    # towards -x direction, and z axis is the up direction.
    init_orientation = pyb.getQuaternionFromEuler([0., 0., 0.])
    return init_orientation

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link masses, last 8 correspond
    to motor masses (24 total).

    Args:
      leg_masses: The leg and motor masses for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of masses is not equal to number
        of links + motors.
    """
    return
    if len(leg_masses) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
      self._pybullet_client.changeDynamics(self.quadruped,
                                           leg_id,
                                           mass=leg_mass)
    motor_masses = leg_masses[len(self._leg_link_ids):]
    for link_id, motor_mass in zip(self._motor_link_ids, motor_masses):
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           mass=motor_mass)

  def SetLegInertias(self, leg_inertias):
    """Set the inertias of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link inertia, last 8 correspond
    to motor inertia (24 total).

    Args:
      leg_inertias: The leg and motor inertias for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of inertias is not equal to
      the number of links + motors or leg_inertias contains negative values.
    """
    return
    if len(leg_inertias) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_inertia in zip(self._leg_link_ids, leg_inertias):
      for inertia_value in leg_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           leg_id,
                                           localInertiaDiagonal=leg_inertia)

    motor_inertias = leg_inertias[len(self._leg_link_ids):]
    for link_id, motor_inertia in zip(self._motor_link_ids, motor_inertias):
      for inertia_value in motor_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           localInertiaDiagonal=motor_inertia)

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).N
      motor_control_mode: A MotorControlMode enum.
    """
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)
    t = super(Aliengo, self).ApplyAction(motor_commands, motor_control_mode)
    return t

  def _ClipMotorCommands(self, motor_commands):
    """Clips motor commands.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).

    Returns:
      Clipped motor commands.
    """

    # clamp the motor command by the joint limit, in case weired things happens
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    motor_commands = np.clip(motor_commands,
                             current_motor_angles - max_angle_change,
                             current_motor_angles + max_angle_change)
    return motor_commands

  @classmethod
  def GetConstants(cls):
    del cls
    return laikago_constants

  def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                              foot_local_position):
    """Use IK to compute the motor angles, given the foot link's local position.

    Args:
      leg_id: The leg index.
      foot_local_position: The foot link's position in the base frame.

    Returns:
      A tuple. The position indices and the angles for all joints along the
      leg. The position indices is consistent with the joint orders as returned
      by GetMotorAngles API.
    """
    assert len(self._foot_link_ids) == self.num_legs
    # toe_id = self._foot_link_ids[leg_id]

    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = foot_position_in_hip_frame_to_joint_angle(
        foot_local_position - HIP_OFFSETS[leg_id],
        l_hip_sign=(-1)**(leg_id + 1))

    # Joint offset is necessary for Laikago.
    joint_angles = np.multiply(
        np.asarray(joint_angles) -
        np.asarray(self._motor_offset)[joint_position_idxs],
        self._motor_direction[joint_position_idxs])

    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles.tolist()

  # def GetFootPositionsInBaseFrame(self):
  #   """Get the robot's foot position in the base frame."""
  #   motor_angles = self.GetMotorAngles()
  #   return foot_positions_in_base_frame(motor_angles)

  def ComputeJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Does not work for Minitaur which has the four bar mechanism for now.
    motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
    return analytical_leg_jacobian(motor_angles, leg_id)
