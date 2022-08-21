import numpy as np
import pybullet
from os.path import exists
from ant_bullet_env.resources.env_bases import MJCFBaseBulletEnv
from ant_bullet_env.resources.scene_stadium import SinglePlayerStadiumScene
from ant_bullet_env.resources.urdf_loader import URDFLoader
import keyboard
import cv2
import termcolor

def solve_maze(file_path, offset=20, check = 20, debug=False):
    image = cv2.imread(file_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y = img.shape
    points = []
    print(x, y)
    current_x, current_y = offset, 5
    direction = 'down'
    done = False
    min_distance = 6
    
    for i in range(204):
        if debug:
          print('-'*100)
          print(i)
          cv2.circle(image, (current_y, current_x), 3, (0, 255, 0), 1)
          cv2.imshow('test', image)
          key = cv2.waitKey()
          if key == ord('q') or key == 27:
              break
        points.append((-current_y, current_x))
        
        if direction == 'down':
            if debug:
              print("turn left or not", img[current_x, current_y:current_y+check])
            if sum(img[current_x, current_y:current_y+check]) > 0:
                if debug:
                  print("direct or not", img[current_x:current_x+check, current_y])
                if sum(img[current_x:current_x+check, current_y]) == 0:
                    current_x += offset
                else:
                    direction = 'right'
            else:
                direction = 'left'
                current_y += offset

        elif direction == 'left':
            if debug:
              print("turn left or not", img[current_x-check:current_x, current_y])
            if sum(img[current_x-check:current_x, current_y]) > 0:
                if debug:
                  print("direct or not", img[current_x, current_y:current_y+check])
                if sum(img[current_x, current_y:current_y+check]) == 0:
                    current_y += offset
                else:
                    direction = 'down'
            else:
                direction = 'up'
                current_x -= offset

        elif direction == 'up':
            if debug:
              print("turn left or not", img[current_x, current_y-check:current_y])
            if sum(img[current_x, current_y-check:current_y]) > 0:
                if debug:
                  print("direct or not", img[current_x-check:current_x, current_y])
                if sum(img[current_x-check:current_x, current_y]) == 0:
                    current_x -= offset
                else:
                    direction = 'left'
                    # current_y += offset
            else:
                direction = 'right'
                current_y -= offset

        elif direction == 'right':
            if debug:
              print("turn left or not", img[current_x:current_x+check, current_y])
            if sum(img[current_x:current_x+check, current_y]) > 0:
                if debug:
                  print("direct or not", img[current_x, current_y-check:current_y])
                if sum(img[current_x, current_y-check:current_y]) == 0:
                    current_y -= offset
                else:
                    direction = 'up'
            else:
                direction = 'down'
                current_x += offset
        
        if debug:
          print(direction)

        if sum(img[current_x-min_distance:current_x, current_y]) > 0:
            current_x += min_distance
        if sum(img[current_x:current_x+min_distance, current_y]) > 0:
            current_x -= min_distance
        if sum(img[current_x, current_y-min_distance:current_y]) > 0:
            current_y += min_distance
        if sum(img[current_x, current_y:current_y+min_distance]) > 0:
            current_y -= min_distance
    return points 


class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

  def __init__(self, robot, render, random, max_goal_dist=90, min_dist=5, max_dist=40, goal_from_keyboard=False):
    # print("WalkerBase::__init__ start")
    self.camera_x = 0
    self.walk_target_x, self.walk_target_y = self.robot.walk_target_x, self.robot.walk_target_y  # kilometer away
    self.stateId = -1
    self.cycle = 0
    self.random = random
    self.max_goal_dist = max_goal_dist
    self.min_dist = min_dist
    self.max_dist = max_dist
    if not random and not goal_from_keyboard:
      self.coordinates = solve_maze('C:/Users/Parham/Downloads/Project/Maze_Solver/test2.png')
    MJCFBaseBulletEnv.__init__(self, robot, render)
    self.times_we_got_it = 0
    self.goal_from_keyboard = goal_from_keyboard
    self.chooseGoal(render=False)
    if self.goal_from_keyboard:
      keyboard.add_hotkey('up', self.chooseGoal, args=(True, False, 'up'))
      keyboard.add_hotkey('down', self.chooseGoal, args=(True, False, 'down'))
      keyboard.add_hotkey('right', self.chooseGoal, args=(True, False, 'right'))
      keyboard.add_hotkey('left', self.chooseGoal, args=(True, False, 'left'))

  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene


  def reset(self):
    # if (self.stateId >= 0):
    #   print("restoreState self.stateId:",self.stateId)
    #   self._p.restoreState(self.stateId)

    r = MJCFBaseBulletEnv.reset(self)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
    self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)

    return r

  def _isDone(self):
    return self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
    "Used by multiplayer stadium to move sideways, to another running lane."
    self.cpp_robot.query_position()
    pose = self.cpp_robot.root_part.pose()
    pose.move_xyz(
        init_x, init_y, init_z
    )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
    self.cpp_robot.set_pose(pose)

  electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints

  def getNextGoalCoordinates(self, from_file=False):
    # from_file = from_file and exists('C:\\Users\\Parham\\Downloads\\Project\\Envs\\Ant-Bullet-Env\\ant_bullet_env\\resources\\goals.txt')
    if from_file:
      with open('C:\\Users\\Parham\\Downloads\\Project\\Envs\\Ant-Bullet-Env\\ant_bullet_env\\resources\\goals.txt', 'r') as file:
        list_of_lines = file.readlines()
        # number_of_lines = len(list_of_lines)
        goal_line = list_of_lines[self.times_we_got_it]
        x, y = (int(s) for s in goal_line.split(' ') if s.isdigit())     
    else:
      if not self.random:
        x, y = self.coordinates[self.cycle]
        self.cycle += 1
        self.cycle %= len(self.coordinates)
      else:
        x = (np.random.uniform(self.min_dist, self.max_dist) if np.random.randint(2) else
              np.random.uniform(-self.min_dist, -self.max_dist))
        y = (np.random.uniform(self.min_dist, self.max_dist) if np.random.randint(2) else
              np.random.uniform(-self.min_dist, -self.max_dist))
        # choice = np.random.randint(4)
        # print(termcolor.colored('Choice: '+str(choice), 'yellow', attrs=['bold']))
        # if choice % 2 == 0:
        #   y = 0
        #   x = 1e3 if choice == 0 else -1e3
        # else: 
        #   x = 0
        #   y = 1e3 if choice == 1 else -1e3
    return x, y
  
  def chooseGoal(self, render=True, from_file=False, direction=''):
    # print(f"{self.times_we_got_it}-Yeah we got it")
    # print("*"*20)
    change = True
    direction_dist = 7
    if not self.goal_from_keyboard:
      x, y = self.getNextGoalCoordinates(from_file=from_file)
    else:
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
    if change:        
      self.walk_target_x, self.walk_target_y = (x, y)
      # self.times_we_got_it += 1
      self.robot.walk_target_x, self.robot.walk_target_y = (x, y)
      if render:
        URDFLoader(self._p._client, address='./statics/simplegoal.urdf', base=(x, y, 0))
    # print(f"{self.times_we_got_it}-produced")

  
  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)
    if(np.abs(self.potential) < self.max_goal_dist):
      self.chooseGoal()
      self.walk_target_x, self.walk_target_y = self.robot.walk_target_x, self.robot.walk_target_y

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
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

    self.rewards = [
        self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
    ]
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
