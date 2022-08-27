from parham_envs.resources.robot_bases import MJCFWalkerBase, URDFWalkerBase
from parham_envs.resources.robot_locomotors import WalkerBaseBulletEnv
import pybullet_data
import os
import argparse
##########################################ROBOTS#####################################################


class Ant(MJCFWalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    
    def __init__(self):
      
        # load_terrain = bool(input('Load Terrain: '))
        # if load_terrain:
        #   terrain_source = input('Terrain Source: ')
        #   terrain_source = terrain_source if terrain_source else 'csv'
        #   terrain_id = input('Terrain ID: ')
        #   terrain_id = terrain_id if terrain_id else 'mounts'
        #   columns = input('Columns: ')
        #   columns = int(columns) if columns else 256
        #   rows = input('Rows: ')
        #   rows = int(rows) if rows else 256
        #   width = input('Width: ')
        #   width = float(width) if width else 0.5
        #   length = input('Length: ')
        #   length = float(length) if length else 0.5
        #   height = input('Height: ')
        #   height = float(height) if height else 0.5
        # else:
        #   terrain_source, terrain_id, columns, rows, width, length, height = 'png', 'maze', 256, 256, 0.5, 0.5, 2

        load_terrain, terrain_source, terrain_id, columns, rows, width, length, height = 0, 'csv', 'mounts', 256, 256, 0.5, 0.5, 0.5
        MJCFWalkerBase.__init__(self, 'ant.xml', "torso", action_dim=8, obs_dim=28, power=2.5, load_terrain=load_terrain,
                                 terrain_source=terrain_source, terrain_id=terrain_id, columns=columns, rows=rows, width=width, length=length, height=height)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class Laikago(URDFWalkerBase):
    foot_list = ['FR_lower_leg', 'FL_lower_leg', 'RR_lower_leg', 'RL_lower_leg']
    
    def __init__(self):
        URDFWalkerBase.__init__(self, os.path.join(pybullet_data.getDataPath(), "laikago/laikago.urdf"), "torso", action_dim=12, obs_dim=40, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class Rex(URDFWalkerBase):
    foot_list = []
    
    def __init__(self):
        URDFWalkerBase.__init__(self, 'C:\\Users\\Parham\\Downloads\\Project_deeprl\\Parham-Envs\\parham_envs\\resources\\statics\\urdf\\urdf\\rex.urdf', "torso", positional_orientation=[0, 0, 1, 0], action_dim=22, obs_dim=40, power=.02)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class Quadruped(MJCFWalkerBase):
    
    def __init__(self):
        MJCFWalkerBase.__init__(self, os.path.join(pybullet_data.getDataPath(), 'quadrapad/quadrapad.xml'), "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


##########################################################################################################

#############################################Main_Bullet_Envs########################################################


class AntBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=True):
    self.robot = Ant()
    WalkerBaseBulletEnv.__init__(self, self.robot, render, training=False, random=True, goal_from_keyboard=False, max_goal_dist=60, min_dist=5, max_dist=15)


class TrainingAntBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Ant()
    WalkerBaseBulletEnv.__init__(self, self.robot, render, training=True)


class LaikagoBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Laikago()
    WalkerBaseBulletEnv.__init__(self, self.robot, render)

class RexBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Rex()
    WalkerBaseBulletEnv.__init__(self, self.robot, render)


class QuadrupedBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Quadruped()
    WalkerBaseBulletEnv.__init__(self, self.robot, render)


#####################################################################################################################
