from ant_bullet_env.resources.robot_bases import WalkerBase
from ant_bullet_env.resources.robot_locomotors import WalkerBaseBulletEnv

##########################################ROBOTS#####################################################

class Ant(WalkerBase):
    # foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']    
    foot_list = ['FR_upper_leg', 'FR_lower_leg', 'FL_upper_leg', 'FL_lower_leg', 'RR_upper_leg', 'RR_lower_leg', 'RL_upper_leg', 'RL_lower_leg']    
    def __init__(self):
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=12, obs_dim=28, power=2.5, load_terrain=False)
        # WalkerBase.__init__(self, "laikago/laikago.urdf", "torso", action_dim=8, obs_dim=28, power=2.5, load_terrain=False)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

##########################################################################################################

#############################################Main_Bullet_Envs########################################################

class AntBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Ant()
    WalkerBaseBulletEnv.__init__(self, self.robot, render, random=True, max_goal_dist=90, min_dist=10, max_dist=25, goal_from_keyboard=False)

#####################################################################################################################
