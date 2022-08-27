import pybullet
import os


class MJCFLoader:
    def __init__(self, _p, self_collision, address):
        f_name = os.path.join(os.path.dirname(__file__), address)
        if self_collision:
            self.objects = _p.loadMJCF(mjcfFileName=f_name, physicsClientId=_p._client,
                                            flags=pybullet.URDF_USE_SELF_COLLISION |
                                            pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                            pybullet.URDF_GOOGLEY_UNDEFINED_COLORS)
        else:
            self.objects = _p.loadMJCF(mjcfFileName=f_name, physicsClientId=_p._client, flags = pybullet.URDF_GOOGLEY_UNDEFINED_COLORS)
        
