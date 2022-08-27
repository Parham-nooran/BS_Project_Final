import numpy as np
import cv2
import pybullet
from .env_bases import MJCFBaseBulletEnv
from .scene_stadium import SinglePlayerStadiumScene
from .urdf_loader import URDFLoader
from .maze import Maze, MazeManager
import keyboard

class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

  def __init__(self, robot, render=False, goal_from_keyboard=False, random=True, min_dist=5,
               max_dist=25, max_goal_dist=20, training=True):
    # print("WalkerBase::__init__ start")
    self.camera_x = 0
    self.walk_target_x = 1
    self.walk_target_y = 0
    self.stateId = -1
    self.cycle = 0
    self.goal_from_keyboard = goal_from_keyboard
    self.random = random
    self.min_dist = min_dist
    self.max_dist = max_dist
    self.max_goal_dist = max_goal_dist
    self.training = training
    if not training:
      if self.goal_from_keyboard:
        keyboard.add_hotkey('up', self.chooseGoal, args=(True, 'up'))
        keyboard.add_hotkey('down', self.chooseGoal, args=(True, 'down'))
        keyboard.add_hotkey('right', self.chooseGoal, args=(True, 'right'))
        keyboard.add_hotkey('left', self.chooseGoal, args=(True, 'left'))
      elif not random:
        cost_coefficient = 45

        manager = MazeManager()
        maze = manager.add_maze(10, 10)

        maze2 = Maze(10, 10)
        maze2 = manager.add_existing_maze(maze2)
        
        maze_binTree = Maze(10, 10, algorithm = "bin_tree")
        maze_binTree = manager.add_existing_maze(maze_binTree)
        manager.solve_maze(maze.id, "DepthFirstBacktracker")
        manager.set_filename("maze")

        manager.show_maze(maze.id)
        img = cv2.imread('parham-envs/parham_envs/resources/statics/maze.png', cv2.IMREAD_GRAYSCALE)
        img[img>125] = 255
        img[img<=125] = 0
        img = 255-img
        img = img[80:-70, 80:-60]
        cv2.imwrite('parham-envs/parham_envs/resources/statics/maze.png', img)
        self.coordinates = [(-cost_coefficient*y, -cost_coefficient*x) for (x, y), _ in maze.solution_path]
        print(self.coordinates)
        # self.coordinates = solve_maze('C:/Users/Parham/Downloads/Project/Maze_Solver/test2.png')

    MJCFBaseBulletEnv.__init__(self, robot, render)


  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene

  def reset(self, is_first=False):
    if (self.stateId >= 0) and self.training:
      # print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)
    
    r = MJCFBaseBulletEnv.reset(self)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
    self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      # print("saving state self.stateId:",self.stateId)
    
    if not is_first and self.training:
      self.chooseGoal()

    return r

  def _isDone(self):
    return self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
    self.cpp_robot.query_position()
    pose = self.cpp_robot.root_part.pose()
    pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
    self.cpp_robot.set_pose(pose)

  electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints


  def getNextGoalCoordinates(self):
    if not self.random:
        x, y = self.coordinates[self.cycle]
        self.cycle += 1
        self.cycle %= len(self.coordinates)
    elif self.training:
      x = np.random.uniform(0, 1000)
      x = -x if np.random.randint(2) else x
      y = np.sqrt(1e6-x**2)
      y = -y if np.random.randint(2) else y
      print('-'*100)
      print("x: ", x, ", y: ", y, ", sqrt(x^2+y^2): ", np.sqrt(x**2+y**2))
    else:
      x = (np.random.uniform(self.min_dist, self.max_dist) if np.random.randint(2) else
              np.random.uniform(-self.min_dist, -self.max_dist))
      y = (np.random.uniform(self.min_dist, self.max_dist) if np.random.randint(2) else
            np.random.uniform(-self.min_dist, -self.max_dist))
    
    return x, y

  def chooseGoal(self, render=True, direction=''):
    change = True
    direction_dist = 7

    if not self.training and self.goal_from_keyboard:
      if direction == 'up':
        x, y = self.robot.body_xyz[0] - direction_dist, self.robot.body_xyz[1]
      elif direction == 'down':
        x, y = self.robot.body_xyz[0] + direction_dist, self.robot.body_xyz[1]
      elif direction == 'left':
        x, y = self.robot.body_xyz[0], self.robot.body_xyz[1] - direction_dist
      elif direction == 'right':
        x, y = self.robot.body_xyz[0], self.robot.body_xyz[1] + direction_dist
      else:
        change = False
    else:
      x, y = self.getNextGoalCoordinates()
    
    if change:
      self.walk_target_x, self.walk_target_y = (x, y)
      self.robot.walk_target_x, self.robot.walk_target_y = (x, y)
      if render and not self.training:
        URDFLoader(self._p._client, address='./statics/simplegoal.urdf', base=(x, y, 0))
  


  def step(self, a):
    if not self.scene.multiplayer:  
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)
    
    if(not self.training and np.abs(self.potential) < self.max_goal_dist):
      self.chooseGoal()
      self.walk_target_x, self.walk_target_y = self.robot.walk_target_x, self.robot.walk_target_y
    
    feet_collision_cost = 0.0
    for i, f in enumerate(self.robot.feet):

      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      if (self.ground_ids & contact_ids):
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
    ))  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
    debugmode = 0
    if (debugmode):
      print("alive=")
      print(self._alive)
      print("progress")
      print(progress)
      print("electricity_cost")
      print(electricity_cost)
      print("joints_at_limit_cost")
      print(joints_at_limit_cost)
      print("feet_collision_cost")
      print(feet_collision_cost)

    self.rewards = [self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost]

    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)

    return state, sum(self.rewards), bool(done), {}

  def camera_adjust(self):
    x, y, z = self.robot.body_real_xyz

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)

