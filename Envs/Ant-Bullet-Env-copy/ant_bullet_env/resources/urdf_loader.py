import pybullet
import os


class URDFLoader:
    def __init__(self, client, address, useFixedBase=False, base=(0, 0)):
        f_name = os.path.join(os.path.dirname(__file__), address)
        self.objects = pybullet.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], base[2]], useFixedBase=useFixedBase, 
                   physicsClientId=client,
                                        flags=pybullet.URDF_USE_SELF_COLLISION |
                                        pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                        pybullet.URDF_GOOGLEY_UNDEFINED_COLORS)
