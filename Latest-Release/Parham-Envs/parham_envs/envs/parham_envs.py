from parham_envs.resources.robot_bases import MJCFWalkerBase, URDFWalkerBase
from parham_envs.resources.robot_locomotors import WalkerBaseBulletEnv

##########################################ROBOTS#####################################################


class Ant(MJCFWalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    
    def __init__(self):
        MJCFWalkerBase.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class Laikago(URDFWalkerBase):
    foot_list = ['FR_lower_leg', 'FL_lower_leg', 'RR_lower_leg', 'RL_lower_leg']
    
    def __init__(self):
        URDFWalkerBase.__init__(self, "laikago/laikago.urdf", "torso", action_dim=12, obs_dim=40, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class Quadruped(MJCFWalkerBase):
    
    def __init__(self):
        MJCFWalkerBase.__init__(self, "quadruped.xml", "torso", action_dim=8, obs_dim=28, power=2.5)

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



class QuadrupedBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Quadruped()
    WalkerBaseBulletEnv.__init__(self, self.robot, render)


#####################################################################################################################
