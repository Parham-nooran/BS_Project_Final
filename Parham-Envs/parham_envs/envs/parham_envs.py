from parham_envs.resources.robot_bases import MJCFWalkerBase, URDFWalkerBase
from parham_envs.resources.robot_locomotors import WalkerBaseBulletEnv
import pybullet_data
import os
##########################################ROBOTS#####################################################


class Ant(MJCFWalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    
    def __init__(self):
        MJCFWalkerBase.__init__(self, os.path.join(pybullet_data.getDataPath(), "mjcf", 'ant.xml'), "torso", action_dim=8, obs_dim=28, power=2.5)

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

  def __init__(self, render=False):
    self.robot = Ant()
    WalkerBaseBulletEnv.__init__(self, self.robot, render)



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
