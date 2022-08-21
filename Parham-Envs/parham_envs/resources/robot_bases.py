import pybullet
import gym, gym.spaces, gym.utils
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data

class XmlBasedRobot:

  self_collision = True

  def __init__(self, robot_name, action_dim, obs_dim, self_collision):
    self.parts = None
    self.objects = []
    self.jdict = None
    self.ordered_joints = None
    self.robot_body = None

    high = np.ones([action_dim], dtype=np.float32)
    self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
    high = np.inf * np.ones([obs_dim], dtype=np.float32)
    self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    #self.model_xml = model_xml
    self.robot_name = robot_name
    self.self_collision = self_collision

  def addToScene(self, bullet_client, bodies):
    self._p = bullet_client

    if self.parts is not None:
      parts = self.parts
    else:
      parts = {}

    if self.jdict is not None:
      joints = self.jdict
    else:
      joints = {}

    if self.ordered_joints is not None:
      ordered_joints = self.ordered_joints
    else:
      ordered_joints = []

    if np.isscalar(bodies):
      bodies = [bodies]

    dump = 0
    for i in range(len(bodies)):
      if self._p.getNumJoints(bodies[i]) == 0:
        part_name, robot_name = self._p.getBodyInfo(bodies[i])
        self.robot_name = robot_name.decode("utf8")
        part_name = part_name.decode("utf8")
        parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
      for j in range(self._p.getNumJoints(bodies[i])):
        self._p.setJointMotorControl2(bodies[i],
                                      j,
                                      pybullet.POSITION_CONTROL,
                                      positionGain=0.1,
                                      velocityGain=0.1,
                                      force=0)
        jointInfo = self._p.getJointInfo(bodies[i], j)
        joint_name = jointInfo[1]
        part_name = jointInfo[12]

        joint_name = joint_name.decode("utf8")
        part_name = part_name.decode("utf8")

        if dump: print("ROBOT PART '%s'" % part_name)
        if dump:
          print(
              "ROBOT JOINT '%s'" % joint_name
          )  # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )

        parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

        if part_name == self.robot_name:
          self.robot_body = parts[part_name]

        if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
          parts[self.robot_name] = BodyPart(self._p, self.robot_name, bodies, 0, -1)
          self.robot_body = parts[self.robot_name]

        if joint_name[:6] == "ignore":
          Joint(self._p, joint_name, bodies, i, j).disable_motor()
          continue

        if joint_name[:8] != "jointfix":
          joints[joint_name] = Joint(self._p, joint_name, bodies, i, j)
          ordered_joints.append(joints[joint_name])

          joints[joint_name].power_coef = 100.0


    return parts, joints, ordered_joints, self.robot_body

  def reset_pose(self, position, orientation):
    self.parts[self.robot_name].reset_pose(position, orientation)


class MJCFBasedRobot(XmlBasedRobot):

  def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True):
    XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
    self.model_xml = model_xml
    self.doneLoading = 0

  def reset(self, bullet_client):

    self._p = bullet_client
    if (self.doneLoading == 0):
      self.ordered_joints = []
      self.doneLoading = 1
      if self.self_collision:
        self.objects = self._p.loadMJCF(self.model_xml,
                                        flags=pybullet.URDF_USE_SELF_COLLISION |
                                        pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                        pybullet.URDF_GOOGLEY_UNDEFINED_COLORS )
      else:
        self.objects = self._p.loadMJCF(
            os.path.join(pybullet_data.getDataPath(), "mjcf", self.model_xml, flags = pybullet.URDF_GOOGLEY_UNDEFINED_COLORS))
      
      self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
          self._p, self.objects)
    self.robot_specific_reset(self._p)

    s = self.calc_state(
    )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

    return s

  def calc_potential(self):
    return 0

class URDFBasedRobot(XmlBasedRobot):

  def __init__(self, model_xml, robot_name, action_dim, obs_dim, positional_orientation=[0, .5, .5, 0], self_collision=True):
    XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
    self.positional_orientation = positional_orientation
    self.model_xml = model_xml
    self.doneLoading = 0

  def reset(self, bullet_client):

    self._p = bullet_client
    if (self.doneLoading == 0):
      self.ordered_joints = []
      self.doneLoading = 1

      quadruped = self.objects = self._p.loadURDF(self.model_xml, [0,0,.5], self.positional_orientation, useFixedBase=False, 
                                        flags=pybullet.URDF_USE_SELF_COLLISION |
                                        pybullet.URDF_GOOGLEY_UNDEFINED_COLORS )
      
      lower_legs = [2,5,8,11]
      for l0 in lower_legs:
        for l1 in lower_legs:
          if (l1>l0):
            enableCollision = 1
            print("collision for pair", l0, l1, pybullet.getJointInfo(quadruped,l0)[12],
                   pybullet.getJointInfo(quadruped,l1)[12], "enabled=", enableCollision)
            pybullet.setCollisionFilterPair(quadruped, quadruped, l0,l1,enableCollision)
      

      self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
          self._p, self.objects)
    self.robot_specific_reset(self._p)

    s = self.calc_state(
    )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

    return s

  def calc_potential(self):
    return 0

class Pose_Helper:  # dummy class to comply to original interface

  def __init__(self, body_part):
    self.body_part = body_part

  def xyz(self):
    return self.body_part.current_position()

  def rpy(self):
    return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

  def orientation(self):
    return self.body_part.current_orientation()



class URDFWalkerBase(URDFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power, positional_orientation):
    URDFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim, positional_orientation)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    for j in self.ordered_joints:
      j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt

class MJCFWalkerBase(MJCFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power):
    MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    for j in self.ordered_joints:
      j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt

class BodyPart:

  def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
    self.bodies = bodies
    self._p = bullet_client
    self.bodyIndex = bodyIndex
    self.bodyPartIndex = bodyPartIndex
    self.initialPosition = self.current_position()
    self.initialOrientation = self.current_orientation()
    self.bp_pose = Pose_Helper(self)

  def state_fields_of_pose_of(
      self, body_id,
      link_id=-1):  # a method you will most probably need a lot to get pose and orientation
    if link_id == -1:
      (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
    else:
      (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
    return np.array([x, y, z, a, b, c, d])

  def get_position(self):
    return self.current_position()

  def get_pose(self):
    return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

  def speed(self):
    if self.bodyPartIndex == -1:
      (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
    else:
      (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
          self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
    return np.array([vx, vy, vz])

  def current_position(self):
    return self.get_pose()[:3]

  def current_orientation(self):
    return self.get_pose()[3:]

  def get_orientation(self):
    return self.current_orientation()

  def reset_position(self, position):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position,
                                            self.get_orientation())

  def reset_orientation(self, orientation):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(),
                                            orientation)

  def reset_velocity(self, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]):
    self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

  def reset_pose(self, position, orientation):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

  def pose(self):
    return self.bp_pose

  def contact_list(self):
    return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:

  def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
    self.bodies = bodies
    self._p = bullet_client
    self.bodyIndex = bodyIndex
    self.jointIndex = jointIndex
    self.joint_name = joint_name

    jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
    self.lowerLimit = jointInfo[8]
    self.upperLimit = jointInfo[9]

    self.power_coeff = 0

  def set_state(self, x, vx):
    self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

  def current_position(self):  # just some synonyme method
    return self.get_state()

  def current_relative_position(self):
    pos, vel = self.get_state()
    pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
    return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), 0.1 * vel)

  def get_state(self):
    x, vx, _, _ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
    return x, vx

  def get_position(self):
    x, _ = self.get_state()
    return x

  def get_orientation(self):
    _, r = self.get_state()
    return r

  def get_velocity(self):
    _, vx = self.get_state()
    return vx

  def set_position(self, position):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  pybullet.POSITION_CONTROL,
                                  targetPosition=position)

  def set_velocity(self, velocity):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  pybullet.VELOCITY_CONTROL,
                                  targetVelocity=velocity)

  def set_motor_torque(self, torque):  # just some synonyme method
    self.set_torque(torque)

  def set_torque(self, torque):
    self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex],
                                  jointIndex=self.jointIndex,
                                  controlMode=pybullet.TORQUE_CONTROL,
                                  force=torque)  #, positionGain=0.1, velocityGain=0.1)

  def reset_current_position(self, position, velocity):  # just some synonyme method
    self.reset_position(position, velocity)

  def reset_position(self, position, velocity):
    self._p.resetJointState(self.bodies[self.bodyIndex],
                            self.jointIndex,
                            targetValue=position,
                            targetVelocity=velocity)
    self.disable_motor()

  def disable_motor(self):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  controlMode=pybullet.POSITION_CONTROL,
                                  targetPosition=0,
                                  targetVelocity=0,
                                  positionGain=0.1,
                                  velocityGain=0.1,
                                  force=0)

