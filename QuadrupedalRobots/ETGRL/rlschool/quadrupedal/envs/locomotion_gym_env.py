# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
"""This file implements the locomotion gym env."""
import collections
import random as rd
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet  # pytype: disable=import-error
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
from rlschool.quadrupedal.envs.utilities.heightfield import HeightField
from rlschool.quadrupedal.robots import robot_config
from rlschool.quadrupedal.envs.sensors import sensor
from rlschool.quadrupedal.envs.sensors import space_utils
from rlschool.quadrupedal.envs.utilities.terrain import generate_terrain, downstair_terrain,upslope_terrain
from rlschool.quadrupedal.envs.utilities.env_utils import flatten_observations
_ACTION_EPS = 0.01
_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000
import copy
import heightfield_terrain as ht

terrain_modes = ["upstair-fix", "upstair-var", "downstair", "slope", "random", "special",
                 "upstair-random", "downstair-random", "downslope-random", "upslope-random",
                 "balance_beam", "cliff", "hurdle", "cave"]

# env vector(7):
upstair = np.array([0,0,1,0,0,0.08,0.25])
downstair = np.array([0,0,0,1,0,0.08,0.25])
upslope = np.array([1,0,0,0,0.34,0,0])
downslope = np.array([0,1,0,0,0.34,0,0])
plane = np.zeros(7)
upstair_downslope = [upstair,downslope,plane,upstair,downslope,plane,upstair,downslope,plane,upstair,downslope,plane,
                    upstair,downslope,plane,upstair,downslope,plane]
upslope_downstair = [upslope,downstair,plane,upslope,downstair,plane,upslope,downstair,plane,upslope,downstair,plane,
                    upslope,downstair,plane,upslope,downstair,plane]
upstair_downstair = [upstair,downstair,plane,upstair,downstair,plane,upstair,downstair,plane,upstair,downstair,plane,
                    upstair,downstair,plane,upstair,downstair,plane,]
upslope_downslope = [upslope,downslope,plane,upslope,downslope,plane,upslope,downslope,plane,upslope,downslope,plane,
                    upslope,downslope,plane,upslope,downslope,plane,]

random_terrain_dict = {1:upstair_downslope, 2:upslope_downstair, 3:upstair_downstair, 4:upslope_downslope}

class LocomotionGymEnv(gym.Env):
  """The gym environment for the locomotion tasks."""
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 100
  }

  def __init__(self,
               gym_config,
               param=None,
               robot_class=None,
               env_sensors=None,
               robot_sensors=None,
               task=None,
               random=False,
               task_mode="plane",
               hf_terrain_mode="slope",
               height_field_iters=2,
               env_randomizers=None):
    """Initializes the locomotion gym environment.
    Args:
      gym_config: An instance of LocomotionGymConfig.
      robot_class: A class of a robot. We provide a class rather than an
        instance due to hard_reset functionality. Parameters are expected to be
        configured with gin.
      sensors: A list of environmental sensors for observation.
      task: A callable function/class to calculate the reward and termination
        condition. Takes the gym env as the argument when calling.
      env_randomizers: A list of EnvRandomizer(s). An EnvRandomizer may
        randomize the physical property of minitaur, change the terrrain during
        reset(), or add perturbation forces during step().
    Raises:
      ValueError: If the num_action_repeat is less than 1.
    """
    self.env_info = [[-100,100,np.array([1,0,0,0,0,0,0])]]
    self.add_height = 0
    self.robot_pos = [0.0, 0.0, 0.0]
    self.dynamic_param = {}
    self.seed()
    self.stepheight = 0.05
    self.stepwidth = 0.33
    self.stepnum = 50
    self._gym_config = gym_config
    self._robot_class = robot_class
    self._robot_sensors = robot_sensors
    self._param = param
    self._random = random
    self._env_sensors = env_sensors if env_sensors is not None else list()
    if self._robot_class is None:
      raise ValueError('robot_class cannot be None.')

    # A dictionary containing the objects in the world other than the robot.
    self._world_dict = {}
    self._task = task  # task = simple_forward_task.SimpleForwardTask, in "build_regular_env()"
    self.task_mode = task_mode
    self.hf_terrain_mode = hf_terrain_mode
    self.parametric_heightfield = None

    self._env_randomizers = env_randomizers if env_randomizers else []

    # This is a workaround due to the issue in b/130128505#comment5
    if isinstance(self._task, sensor.Sensor):
      self._env_sensors.append(self._task)

    # Simulation related parameters.
    self._num_action_repeat = gym_config.simulation_parameters.num_action_repeat
    self._on_rack = gym_config.simulation_parameters.robot_on_rack
    if self._num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self._sim_time_step = gym_config.simulation_parameters.sim_time_step_s
    self._env_time_step = self._num_action_repeat * self._sim_time_step
    self._env_step_counter = 0

    self._num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS /
                                             self._num_action_repeat)
    self._is_render = gym_config.simulation_parameters.enable_rendering

    # The wall-clock time at which the last frame is rendered.
    self._last_frame_time = 0.0
    self._show_reference_id = -1

    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
      self._pybullet_client.configureDebugVisualizer(
          pybullet.COV_ENABLE_GUI,
          gym_config.simulation_parameters.enable_rendering_gui)
      if hasattr(self._task, '_draw_ref_model_alpha'):
        self._show_reference_id = pybullet.addUserDebugParameter("show reference",0,1,
          self._task._draw_ref_model_alpha)
      self._delay_id = pybullet.addUserDebugParameter("delay",0,0.3,0)
    else:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.DIRECT)
    self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    if gym_config.simulation_parameters.egl_rendering:
      self._pybullet_client.loadPlugin('eglRendererPlugin')

    # The action list contains the name of all actions.
    self._build_action_space()

    # Set the default render options.
    self._camera_dist = gym_config.simulation_parameters.camera_distance
    self._camera_yaw = gym_config.simulation_parameters.camera_yaw
    self._camera_pitch = gym_config.simulation_parameters.camera_pitch
    self._render_width = gym_config.simulation_parameters.render_width
    self._render_height = gym_config.simulation_parameters.render_height
    self._first_reset = True
    self._hard_reset = True
    self.reset()

    self._hard_reset = gym_config.simulation_parameters.enable_hard_reset
    if self.task_mode is "random":
      self._hard_reset = True

    # Construct the observation space from the list of sensors. Note that we
    # will reconstruct the observation_space after the robot is created.
    self.observation_space = (
        space_utils.convert_sensors_to_gym_space_dictionary(
            self.all_sensors()))
    self.hf = HeightField(2)
    # if self.height_field:
    #     # Do 3x for extra roughness
    #     for i in range(height_field_iters):
    #         self.hf._generate_field(self)
    # print("height_field:",self.height_field)

  def _build_action_space(self):
    """Builds action space based on motor control mode."""
    motor_mode = self._gym_config.simulation_parameters.motor_control_mode
    if motor_mode == robot_config.MotorControlMode.HYBRID:
      action_upper_bound = []
      action_lower_bound = []
      action_config = self._robot_class.ACTION_CONFIG
      for action in action_config:
        action_upper_bound.extend([6.28] * 5)
        action_lower_bound.extend([-6.28] * 5)
      self.action_space = spaces.Box(np.array(action_lower_bound),
                                     np.array(action_upper_bound),
                                     dtype=np.float32)
    elif motor_mode == robot_config.MotorControlMode.TORQUE:
      # TODO (yuxiangy): figure out the torque limits of robots.
      torque_limits = np.array([33.5] * len(self._robot_class.ACTION_CONFIG))
      self.action_space = spaces.Box(-torque_limits,
                                     torque_limits,
                                     dtype=np.float32)
    else:
      # Position mode
      action_upper_bound = []
      action_lower_bound = []
      action_config = self._robot_class.ACTION_CONFIG
      for action in action_config:
        action_upper_bound.append(action.upper_bound)
        action_lower_bound.append(action.lower_bound)

      self.action_space = spaces.Box(np.array(action_lower_bound),
                                     np.array(action_upper_bound),
                                     dtype=np.float32)

  def close(self):
    if hasattr(self, '_robot') and self._robot:
      self._robot.Terminate()

  def seed(self, seed=None):
    self.np_random, self.np_random_seed = seeding.np_random(seed)
    return [self.np_random_seed]

  def all_sensors(self):
    """Returns all robot and environmental sensors."""
    return self._robot.GetAllSensors() + self._env_sensors

  def sensor_by_name(self, name):
    """Returns the sensor with the given name, or None if not exist."""
    for sensor_ in self.all_sensors():
      if sensor_.get_name() == name:
        return sensor_
    return None

  def reset(self,**kwargs):
    """Resets the robot's position in the world or rebuild the sim world.
    The simulation world will be rebuilt if self._hard_reset is True.
    Args:
      initial_motor_angles: A list of Floats. The desired joint angles after
        reset. If None, the robot will use its built-in value.
      reset_duration: Float. The time (in seconds) needed to rotate all motors
        to the desired initial values.
      reset_visualization_camera: Whether to reset debug visualization camera on
        reset.
    Returns:
      A numpy array contains the initial observation after reset.
    """
    self.robot_pos = [0.0, 0.0, 0.0]
    for sensor in self._robot_sensors:
      sensor.reset()
    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_RENDERING, 0)
    hard_set = False
    if "hardset" in kwargs.keys():
      if kwargs["hardset"]:
        # self._hard_reset = True
        hard_set = True
    # print("hardset:",self._hard_reset)
    # print("kwargs:",kwargs)
    # t0 = time.clock()
    # Clear the simulation world and rebuild the robot interface.
    if self._first_reset or self._hard_reset or hard_set:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=self._num_bullet_solver_iterations)
      self._pybullet_client.setTimeStep(self._sim_time_step)
      # print("sim_step:",self._sim_time_step)
      self._pybullet_client.setGravity(0, 0, -10)

      # Rebuild the world.
      # self._world_dict = {
      #     "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
      # }
      # self._pybullet_client.changeDynamics(self._world_dict['ground'],-1,lateralFriction=5)

      if self.task_mode == "heightfield":
        if "heightfield_terrain" in kwargs.keys():
          hf_terrain = kwargs["heightfield_terrain"]
        else:
          hf_terrain = self.hf_terrain_mode
        if hf_terrain == "rough":
          self.parametric_heightfield = ht.RandomUniformTerrain(self._pybullet_client, size=10)
          robot_x, robot_y = -3.5, 0.0
          x_range, y_range = (-0.3, 0.3), (-0.1, 0.1)
          x_range = np.array(x_range) + robot_x
          y_range = np.array(y_range) + robot_y
          self.robot_pos = [robot_x, robot_y, self.parametric_heightfield.getMaxHeightInRange(x_range, y_range)[2]]
        elif hf_terrain == "slope":
          self.parametric_heightfield = ht.SlopeTerrain(self._pybullet_client, size=10)
          robot_x, robot_y = -3.5, 0.0
          x_range, y_range = (-0.3, 0.3), (-0.1, 0.1)
          x_range = np.array(x_range) + robot_x
          y_range = np.array(y_range) + robot_y
          self.robot_pos = [robot_x, robot_y, self.parametric_heightfield.getMaxHeightInRange(x_range, y_range)[2]]
        else:
          self.parametric_heightfield = ht.PlainTerrain(self._pybullet_client)
          robot_x, robot_y = 0.0, 0.0
          self.robot_pos = [robot_x, robot_y, 0.0]
      else:
        ground = self._pybullet_client.loadURDF("plane_implicit.urdf")
        self._pybullet_client.changeDynamics(ground, -1, lateralFriction=5)

      # cs = pybullet.createCollisionShape(pybullet.GEOM_BOX,halfExtents=[0.1,0.06,0.4])
      # id = pybullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs,
      #                               basePosition=[0,0,0])
      # pybullet.changeDynamics(id, -1, lateralFriction=5.0)

      # Rebuild the robot
      self._robot = self._robot_class(
        pybullet_client=self._pybullet_client,
        sensors=self._robot_sensors,
        on_rack=self._on_rack,
        action_repeat=self._gym_config.simulation_parameters.
          num_action_repeat,
        time_step=self._gym_config.simulation_parameters.
          sim_time_step_s,
        motor_control_mode=self._gym_config.simulation_parameters.
          motor_control_mode,
        reset_time=self._gym_config.simulation_parameters.
          reset_time,
        enable_clip_motor_commands=self._gym_config.simulation_parameters.
          enable_clip_motor_commands,
        enable_action_filter=self._gym_config.simulation_parameters.
          enable_action_filter,
        enable_action_interpolation=self._gym_config.simulation_parameters.
          enable_action_interpolation,
        allow_knee_contact=self._gym_config.simulation_parameters.
          allow_knee_contact)

    # Using default special terrain
    if self._first_reset:
      if self.task_mode == "stairslope":
        self.add_height,self.env_info = generate_terrain(mode="special", env_vecs=upstair_downslope)
      elif self.task_mode == "stairstair":
        self.add_height,self.env_info = generate_terrain(mode="special", env_vecs=upstair_downstair)
      elif self.task_mode == "slopestair":
        self.add_height,self.env_info = generate_terrain(mode="special", env_vecs=upslope_downstair)
      elif self.task_mode == "slopeslope":
        self.add_height,self.env_info = generate_terrain(mode="special", env_vecs=upslope_downslope)
      elif self.task_mode == "gallop":
        self.add_height,self.env_info = generate_terrain(stepwidth=0.5, mode="gallop")
      elif self.task_mode == "cave":
        self.add_height,self.env_info = generate_terrain(stepheight=0.18, mode="cave")
      elif self.task_mode == "balancebeam":
        self.add_height,self.env_info = generate_terrain(stepwidth=0.05, stepheight=6, mode="balance_beam")
      elif self.task_mode == "highstair":
        self.add_height,self.env_info = generate_terrain(stepwidth=0.4, stepheight=0.13, mode="upstair-var")
    if self.task_mode == "random":
      random_env = random_terrain_dict[rd.randint(1, 4)]
      self.add_height, self.env_info = generate_terrain(mode="special", env_vecs=random_env)

    # Set dynamic params of the robot and simulator.
    self.dynamic_param = self.get_dynamic_param()
    # if "dynamic_param" in kwargs.keys():
    #   self.set_dynamic_param(kwargs["dynamic_param"], self._random)
    # elif self._param is not None:
    #   self.set_dynamic_param(self._param, self._random)

    # Reset the pose of the robot.
    reset_duration = 0.0
    yaw = 0.0
    add_x = 0.0
    if "yaw" in kwargs.keys():
      yaw = kwargs["yaw"]
    if "x_noise" in kwargs.keys() and kwargs["x_noise"]:
      add_x = np.random.uniform(-0.2, 0.1)
    self.robot_pos += np.array([add_x, 0.0, 0.0])
    self.robot_pos += np.array(self._robot.GetDefaultInitPosition())
    self.robot_pos = self.robot_pos.tolist()
    self._robot.Reset(reload_urdf=False,
                      reset_time=reset_duration,
                      default_pose=self.robot_pos,
                      yaw=yaw)

    reset_visualization_camera = True
    if reset_visualization_camera:
      self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist,
                                                       self._camera_yaw,
                                                       self._camera_pitch,
                                                       [0, 0, 0])
    self._env_step_counter = 0
    self._first_reset = False
    self._last_action = np.zeros(self.action_space.shape)

    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_RENDERING, 1)

    for s in self.all_sensors():
      s.on_reset(self)

    if self._task and hasattr(self._task, 'reset'):
      self._task.reset(self)

    # Loop over all env randomizers.
    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_env(self)

    # if self.height_field and not self.first_reset:
    #   self.hf.UpdateHeightField(self)

    info = {}
    info["env_info"] = self.env_info
    info["rot_quat"] = self._robot.GetTrueBaseOrientation()
    info["rot_mat"] = pybullet.getMatrixFromQuaternion(info["rot_quat"])
    info["rot_euler"] = self._robot.GetTrueBaseRollPitchYaw()
    info["drpy"] = self._robot.GetTrueBaseRollPitchYawRate()
    info["base_position"] = self._robot.GetBasePosition()
    info["base_velocity"] = self._robot.GetBaseVelocity()
    info["foot_position"] = self._robot.GetFootPositionsInBaseFrame()
    info["real_contact"] = self._robot.GetFootContacts()
    info["joint_angle"] = self._robot.GetTrueMotorAngles()
    info["joint_torque"] = self._robot.GetTrueMotorTorques()
    info["energy"] = self._robot.GetEnergyConsumptionPerControlStep()
    info["transport_cost"] = self._robot.GetCostOfTransport()
    info["latency"] = self._robot.GetControlLatency()
    info["yaw_init"] = yaw
    info['dynamics'] = np.array(
      [self.dynamic_param['control_latency'], self.dynamic_param["footfriction"], self.dynamic_param['basemass']])

    return self._get_observation(),info

  def step(self, action):
    """Step forward the simulation, given the action.
    Args:
      action: Can be a list of desired motor angles for all motors when the
        robot is in position control mode; A list of desired motor torques. Or a
        list of tuples (q, qdot, kp, kd, tau) for hybrid control mode. The
        action must be compatible with the robot's motor control mode. Also, we
        are not going to use the leg space (swing/extension) definition at the
        gym level, since they are specific to Minitaur.
    Returns:
      observations: The observation dictionary. The keys are the sensor names
        and the values are the sensor readings.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.
    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    self._last_base_position = self._robot.GetBasePosition()
    self._last_action = action
    if self._random:
      control_latency = np.random.uniform(0.01,0.02)
      self._robot.SetControlLatency(control_latency)
      # print("latency:",control_latency)
    # print("latency:",self._robot.GetControlLatency())
    time_spent = time.time() - self._last_frame_time
    self._last_frame_time = time.time()
    # print("[step] time_spent: ", time_spent)
    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_to_sleep = self._env_time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)

      base_pos = self._robot.GetBasePosition()
      # Also keep the previous orientation of the camera set by the user.
      [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
      self._pybullet_client.resetDebugVisualizerCamera(
        dist, yaw, pitch, base_pos)
      self._pybullet_client.configureDebugVisualizer(
        pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
      alpha = 1.
      if self._show_reference_id > 0:
        alpha = self._pybullet_client.readUserDebugParameter(self._show_reference_id)
      
      ref_col = [1, 1, 1, alpha]
      if hasattr(self._task, '_ref_model'):
        self._pybullet_client.changeVisualShape(self._task._ref_model, -1, rgbaColor=ref_col)
        for l in range (self._pybullet_client.getNumJoints(self._task._ref_model)):
        	self._pybullet_client.changeVisualShape(self._task._ref_model, l, rgbaColor=ref_col)
    
      delay = self._pybullet_client.readUserDebugParameter(self._delay_id)
      if (delay>0):
        time.sleep(delay)

    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_step(self)
    # print("step_action:",action)
    # robot class and put the logics here.
    # call the PD controller of joint position, step time: action_repeat * 0.002
    torques = self._robot.Step(action)

    for s in self.all_sensors():
      s.on_step(self)

    if self._task and hasattr(self._task, 'update'):
      self._task.update(self)

    reward = self._reward(action,torques)  #

    done = self._termination()
    self._env_step_counter += 1
    if done:
      self._robot.Terminate()
    
    info = {}
    info["env_info"] = self.env_info
    info["rot_quat"] = self._robot.GetTrueBaseOrientation()
    info["rot_mat"] = pybullet.getMatrixFromQuaternion(info["rot_quat"])
    info["rot_euler"] = self._robot.GetTrueBaseRollPitchYaw()
    info["drpy"] = self._robot.GetTrueBaseRollPitchYawRate()
    info["base_position"] = self._robot.GetBasePosition()
    info["base_velocity"] = self._robot.GetBaseVelocity()
    info["foot_position"] = self._robot.GetFootPositionsInBaseFrame()
    info["foot_position_world"] = self._robot.GetFootPositionsInWorldFrame()
    info["real_contact"] = self._robot.GetFootContacts()
    info["joint_angle"] = self._robot.GetTrueMotorAngles()
    info["joint_torque"] = self._robot.GetTrueMotorTorques()
    info["energy"] = self._robot.GetEnergyConsumptionPerControlStep()
    info["transport_cost"] = self._robot.GetCostOfTransport()
    info["latency"] = self._robot.GetControlLatency()
    return self._get_observation(), reward, done, info

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError('Unsupported render mode:{}'.format(mode))
    base_pos = self._robot.GetBasePosition()

    cameraEyePosition = [0, 0, 2]
    cameraTargetPosition = [0.001, 0, 0]

    cameraUpVector = [0, 0, 1]
    view_matrix = self._pybullet_client.computeViewMatrix(
      cameraEyePosition, cameraTargetPosition, cameraUpVector)

    # view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=base_pos,
    #     distance=self._camera_dist,
    #     yaw=self._camera_yaw,
    #     pitch=self._camera_pitch,
    #     roll=0,
    #     upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(self._render_width) / self._render_height,
        nearVal=0.1,
        farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=self._render_width,
        height=self._render_height,
        renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_terrain_height(self, world_x, world_y):
    if self.parametric_heightfield == None:
      return 0.0
    else:
      return self.parametric_heightfield.getHeight(world_x, world_y)

  def get_ground(self):
    """Get simulation ground model."""
    return self._world_dict['ground']

  def set_ground(self, ground_id):
    """Set simulation ground model."""
    self._world_dict['ground'] = ground_id

  def get_dynamic_param(self):
    if not self.dynamic_param:
      control_latency = 0.002
      footfriction = 1.0
      motor_kp = np.zeros(12)
      motor_kd = np.zeros(12)
      gravity = np.array([0, 0, -10])
      basemass = self._robot.GetBaseMassesFromURDF()[0]
      baseinertia = self._robot.GetBaseInertiasFromURDF()[0]
      legmass = self._robot.GetLegMassesFromURDF()
      leginertia = self._robot.GetLegInertiasFromURDF()
      self.dynamic_param['control_latency'] = control_latency
      self.dynamic_param['footfriction'] = footfriction
      self.dynamic_param['motor_kp'] = motor_kp
      self.dynamic_param['motor_kd'] = motor_kd
      self.dynamic_param['gravity'] = gravity
      self.dynamic_param['basemass'] = basemass
      self.dynamic_param['baseinertia'] = baseinertia
      self.dynamic_param['legmass'] = legmass
      self.dynamic_param['leginertia'] = leginertia
    return self.dynamic_param

  def set_dynamic_param(self, dynamic_param, random):
    control_latency = 0.002
    footfriction = 1.0
    motor_kp = np.zeros(12)
    motor_kd = np.zeros(12)
    gravity = np.array([0, 0, -10])
    basemass = self._robot.GetBaseMassesFromURDF()[0]
    baseinertia = self._robot.GetBaseInertiasFromURDF()[0]
    legmass = self._robot.GetLegMassesFromURDF()
    leginertia = self._robot.GetLegInertiasFromURDF()
    if random is not True:
      if 'control_latency' in dynamic_param.keys():
        control_latency = 0.001 * dynamic_param['control_latency']
      if 'footfriction' in dynamic_param.keys():
        footfriction = dynamic_param['footfriction']
      if 'motor_kp' in dynamic_param.keys() and 'motor_kd' in dynamic_param.keys():
        motor_kp = dynamic_param['motor_kp']
        motor_kd = dynamic_param['motor_kd']
      if 'gravity' in dynamic_param.keys():
        gravity = dynamic_param['gravity']
      if 'basemass' in dynamic_param.keys():
        basemass = [basemass * dynamic_param['basemass']]
      if 'baseinertia' in dynamic_param.keys():
        baseinertia_ratio = dynamic_param['baseinertia']
        baseinertia = [(baseinertia[0] * baseinertia_ratio[0],
                        baseinertia[1] * baseinertia_ratio[1],
                        baseinertia[2] * baseinertia_ratio[2])]
      if 'legmass' in dynamic_param.keys():
        legmass_ratio = dynamic_param['legmass']
        legmass = legmass * np.array([legmass_ratio[0],legmass_ratio[0],legmass_ratio[1],legmass_ratio[2]]*5)
      if 'leginertia' in dynamic_param.keys():
        leginertia_ratio = dynamic_param['leginertia']
        leginertia_new = []
        for i in range(12):
          leginertia_new.append((leginertia_ratio[i] * leginertia[i][0],
                                 leginertia_ratio[i] * leginertia[i][1],
                                 leginertia_ratio[i] * leginertia[i][2]))
        leginertia = copy.deepcopy(leginertia_new)
    else:
      control_latency = np.random.uniform(0.035, 0.045)
      footfriction = np.random.uniform(1, 2)
      motor_kp = np.random.uniform([83, 89, 83, 65, 100, 73, 68, 78, 76, 67, 74, 66],
                                   [109, 109, 109, 109, 106, 97, 90, 80, 100, 99, 80, 109], 12)
      motor_kd = np.random.normal([1.1, 2.9, 2.15, 1.8, 3.19, 1.8, 1.1, 3.99, 1.7, 1.2, 3.99, 2.7],
                                  [1.9, 3.1, 2.85, 2.0, 3.21, 2.2, 1.9, 4.01, 2.1, 2.0, 4.01, 3.9], 12)
      gravity = np.random.uniform([-1, -1, 8], [1, 1, 12], size=3)
      basemass_ratio = np.random.uniform(0.8, 1.2)
      basemass = [basemass * basemass_ratio]
      baseinertia_ratio = np.random.uniform([0.3, 1.3, 0.5], [0.7, 1.7, 1.5], 3)
      baseinertia = [(baseinertia[0] * baseinertia_ratio[0],
                      baseinertia[1] * baseinertia_ratio[1],
                      baseinertia[2] * baseinertia_ratio[2])]
      legmass_ratio = np.random.uniform([1, 1.1, 0.8], [1.4, 1.5, 1.6], 3)
      legmass = legmass * np.array([legmass_ratio[0], legmass_ratio[1], legmass_ratio[2]] * 4)
      leginertia_ratio = np.random.uniform([0.8, 0.55, 0.69, 1.4, 1.33, 0.5, 1.37, 0.48, 1.06, 1.39, 1.4, 1.4],
                                           [1.4, 0.75, 0.99, 1.6, 1.53, 0.8, 1.57, 0.68, 1.46, 1.59, 1.6, 1.6], 12)
      leginertia_new = []
      for i in range(12):
        leginertia_new.append((leginertia_ratio[i] * leginertia[i][0],
                               leginertia_ratio[i] * leginertia[i][1],
                               leginertia_ratio[i] * leginertia[i][2]))
      leginertia = copy.deepcopy(leginertia_new)

    self._robot.SetControlLatency(control_latency)
    self._robot.SetFootFriction(footfriction)
    self._robot.SetBaseMasses(basemass)
    self._robot.SetBaseInertias(baseinertia)
    self._robot.SetLegMasses(legmass)
    self._robot.SetLegInertias(leginertia)
    self._robot.SetMotorGains(motor_kp, motor_kd)
    self._pybullet_client.setGravity(gravity[0], gravity[1], gravity[2])
    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)

    self.dynamic_param['control_latency'] = control_latency
    self.dynamic_param['footfriction'] = footfriction
    self.dynamic_param['motor_kp'] = motor_kp
    self.dynamic_param['motor_kd'] = motor_kd
    self.dynamic_param['gravity'] = gravity
    self.dynamic_param['basemass'] = basemass
    self.dynamic_param['baseinertia'] = baseinertia
    self.dynamic_param['legmass'] = legmass
    self.dynamic_param['leginertia'] = leginertia

  @property
  def rendering_enabled(self):
    return self._is_render

  @property
  def last_base_position(self):
    return self._last_base_position

  @property
  def world_dict(self):
    return self._world_dict.copy()

  @world_dict.setter
  def world_dict(self, new_dict):
    self._world_dict = new_dict.copy()

  def _termination(self):
    if not self._robot.is_safe:
      return True

    if self._task and hasattr(self._task, 'done'):
      return self._task.done(self)

    for s in self.all_sensors():
      s.on_terminate(self)

    return False

  def _reward(self,action,torques):
    if self._task:
      return self._task(self,action,torques)
    return 0

  def _get_observation(self):
    """Get observation of this environment from a list of sensors.
    Returns:
      observations: sensory observation in the numpy array format
    """
    sensors_dict = {}
    for s in self.all_sensors():
      sensors_dict[s.get_name()] = s.get_observation()

    observations = collections.OrderedDict(sorted(list(sensors_dict.items())))
    # print(observations)
    return observations

  def set_time_step(self, num_action_repeat, sim_step=0.001):
    """Sets the time step of the environment.
    Args:
      num_action_repeat: The number of simulation steps/action repeats to be
        executed when calling env.step().
      sim_step: The simulation time step in PyBullet. By default, the simulation
        step is 0.001s, which is a good trade-off between simulation speed and
        accuracy.
    Raises:
      ValueError: If the num_action_repeat is less than 1.
    """
    if num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self._sim_time_step = sim_step
    self._num_action_repeat = num_action_repeat
    self._env_time_step = sim_step * num_action_repeat
    self._num_bullet_solver_iterations = (_NUM_SIMULATION_ITERATION_STEPS /
                                          self._num_action_repeat)
    self._pybullet_client.setPhysicsEngineParameter(
        numSolverIterations=int(np.round(self._num_bullet_solver_iterations)))
    self._pybullet_client.setTimeStep(self._sim_time_step)
    self._robot.SetTimeSteps(self._num_action_repeat, self._sim_time_step)

  def get_time_since_reset(self):
    """Get the time passed (in seconds) since the last reset.
    Returns:
      Time in seconds since the last reset.
    """
    return self._robot.GetTimeSinceReset()

  @property
  def pybullet_client(self):
    return self._pybullet_client

  @property
  def robot(self):
    return self._robot

  @property
  def env_step_counter(self):
    return self._env_step_counter

  @property
  def hard_reset(self):
    return self._hard_reset

  @property
  def last_action(self):
    return self._last_action

  @property
  def env_time_step(self):
    return self._env_time_step

  @property
  def task(self):
    return self._task

  @property
  def robot_class(self):
    return self._robot_class

  @property
  def gym_config(self):
    return self._gym_config

  def get_observation(self):
    return flatten_observations(self._get_observation())