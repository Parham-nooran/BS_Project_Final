# import pybullet as p
# import pybullet_data
# from time import sleep
# import os
# from resources.urdf_loader import URDFLoader

# client = p.connect(p.GUI)
# p.setGravity(0, 0, -10)
# # angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
# # throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)
# # p.loadURDF('simpleplane.urdf')
# # p.loadMJCF(os.path.join(pybullet_data.getDataPath(), "mjcf", 
# #                                                      'ant.xml'),
# #                                         flags=p.URDF_USE_SELF_COLLISION |
# #                                         p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
# #                                         p.URDF_GOOGLEY_UNDEFINED_COLORS)
# # URDFLoader(client, './statics/cube.urdf', base=(0, 0, 5))
# # URDFLoader(client, './statics/torus.urdf', base=(0, 0, 5))
# wheel_indices = [1, 3, 4, 5]
# hinge_indices = [0, 2]

# while True:
#     # user_angle = p.readUserDebugParameter(angle)
#     # user_throttle = p.readUserDebugParameter(throttle)
#     # for joint_index in wheel_indices:
#     #     p.setJointMotorControl2(car, joint_index,
#     #                             p.VELOCITY_CONTROL,
#     #                             targetVelocity=user_throttle)
#     # for joint_index in hinge_indices:
#     #     p.setJointMotorControl2(car, joint_index,
#     #                             p.POSITION_CONTROL, 
#     #                             targetPosition=user_angle)
#     p.stepSimulation()
#     # sleep(0.003)


import gym
import parham_envs
import pybullet
import time

env = gym.make('RexBulletEnv-v0')
env.render()

for i in range(1000):
    env.reset()
    action = env.action_space.sample()
    new_obs, reward, done, info = env.step(action)
    while not done:
        env.render()
        # print(env.robot.body_xyz)
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        time.sleep(0.001)



# import pybullet as p
# import time
# import pybullet_data
# import os

# p.connect(p.GUI)
# p.setGravity(0,0,-9.8)
# p.setTimeStep(1./500)
# plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'plane.urdf'))
# #p.setDefaultContactERP(0)
# #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS 
# urdfFlags = p.URDF_USE_SELF_COLLISION
# quadruped = p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'laikago/laikago.urdf'), [0,0,.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)

# #enable collision between lower legs

# for j in range (p.getNumJoints(quadruped)):
# 		print(p.getJointInfo(quadruped,j))

# #2,5,8 and 11 are the lower legs
# lower_legs = [2,5,8,11]
# for l0 in lower_legs:
# 	for l1 in lower_legs:
# 		if (l1>l0):
# 			enableCollision = 1
# 			print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
# 			p.setCollisionFilterPair(quadruped, quadruped, l0,l1,enableCollision)

# jointIds=[]
# paramIds=[]
# jointOffsets=[]
# jointDirections=[-1,1,1,1,1,1,-1,1,1,1,1,1]
# jointAngles=[0,0,0,0,0,0,0,0,0,0,0,0]

# for i in range (4):
# 	jointOffsets.append(0)
# 	jointOffsets.append(-0.7)
# 	jointOffsets.append(0.7)

# maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

# for j in range (p.getNumJoints(quadruped)):
#         p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
#         info = p.getJointInfo(quadruped,j)
#         #print(info)
#         jointName = info[1]
#         jointType = info[2]
#         if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
#                 jointIds.append(j)

		
# p.getCameraImage(480,320)
# p.setRealTimeSimulation(0)

# joints=[]

# with open("data1.txt","r") as filestream:
# 	for line in filestream:
# 		maxForce = p.readUserDebugParameter(maxForceId)
# 		currentline = line.split(",")
# 		frame = currentline[0]
# 		t = currentline[1]
# 		joints=currentline[2:14]
# 		for j in range (12):
# 			targetPos = float(joints[j])
# 			p.setJointMotorControl2(quadruped,jointIds[j],p.POSITION_CONTROL,jointDirections[j]*targetPos+jointOffsets[j], force=maxForce)
# 		p.stepSimulation()
# 		for lower_leg in lower_legs:
# 			#print("points for ", quadruped, " link: ", lower_leg)
# 			pts = p.getContactPoints(quadruped,-1, lower_leg)
# 			#print("num points=",len(pts))
# 			#for pt in pts:
# 			#	print(pt[9])
# 		time.sleep(1./500.)


# index = 0
# for j in range (p.getNumJoints(quadruped)):
#         p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
#         info = p.getJointInfo(quadruped,j)
#         js = p.getJointState(quadruped,j)
#         #print(info)
#         jointName = info[1]
#         jointType = info[2]
#         if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
#                 paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,(js[0]-jointOffsets[index])/jointDirections[index]))
#                 index=index+1


# p.setRealTimeSimulation(1)

# while (1):
	
# 	for i in range(len(paramIds)):
# 		c = paramIds[i]
# 		targetPos = p.readUserDebugParameter(c)
# 		maxForce = p.readUserDebugParameter(maxForceId)
# 		p.setJointMotorControl2(quadruped,jointIds[i],p.POSITION_CONTROL,jointDirections[i]*targetPos+jointOffsets[i], force=maxForce)
	


# from msvcrt import getch
# while True:
#     key = ord(getch())
#     if key == 27: #ESC
#         print('esc\nexit')
#         break
#     elif key == 13: #Enter
#         print('Enter')
#     elif key == 224: #Special keys (arrows, f keys, ins, del, etc.)
#         key = ord(getch())
#         if key == 80: #Down arrow
#             print('Arrow down')
#         elif key == 72: #Up arrow
#             print('Arrow up')
#         elif key == 77:
#             print('Arrow right')
#         elif key == 75:
#             print('Arrow left')
            
#         else:
#             print(key)


    