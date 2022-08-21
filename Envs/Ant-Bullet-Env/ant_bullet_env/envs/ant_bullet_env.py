from ant_bullet_env.resources.robot_bases import WalkerBase
from ant_bullet_env.resources.robot_locomotors import WalkerBaseBulletEnv

##########################################ROBOTS#####################################################

class Ant(WalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    # foot_list = ['hip_front_left', 'hip_front_right', 'hip_left_back', 'hip_right_back']
    
    def __init__(self):
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5, load_terrain=True)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

##########################################################################################################

#############################################Main_Bullet_Envs########################################################


class AntBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Ant()
    WalkerBaseBulletEnv.__init__(self, self.robot, render, random=False, max_goal_dist=90, min_dist=5, max_dist=15, goal_from_keyboard=False)

#####################################################################################################################
