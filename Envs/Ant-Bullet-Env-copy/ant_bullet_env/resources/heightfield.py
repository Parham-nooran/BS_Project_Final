import pybullet as p
import pybullet_data as pd
import math
import time

textureId = -1

useProgrammatic = 0
useTerrainFromPNG = 1
useDeepLocoCSV = 2
updateHeightfield = False

heightfieldSource = useTerrainFromPNG
numHeightfieldRows = 256
numHeightfieldColumns = 256
import random
random.seed(10)


class HeightField():
    def __init__(self):
        self.hf_id = 0
        self.terrainShape = 0
        self.heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

    def _generate_field(self, client, heightPerturbationRange=0.08):

        client.setAdditionalSearchPath(pd.getDataPath())

        client.configureDebugVisualizer(
            client.COV_ENABLE_RENDERING, 0)
        heightPerturbationRange = heightPerturbationRange
        if heightfieldSource == useProgrammatic:
            for j in range(int(numHeightfieldColumns / 2)):
                for i in range(int(numHeightfieldRows / 2)):
                    height = random.uniform(0, heightPerturbationRange)
                    self.heightfieldData[2 * i +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + (2 * j + 1) *
                                         numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 + (2 * j + 1) *
                                         numHeightfieldRows] = height

            terrainShape = client.createCollisionShape(
                shapeType=client.GEOM_HEIGHTFIELD,
                meshScale=[1., 1., 5],
                heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                heightfieldData=self.heightfieldData,
                numHeightfieldRows=numHeightfieldRows,
                numHeightfieldColumns=numHeightfieldColumns)
            terrain = client.createMultiBody(0, terrainShape)
            client.resetBasePositionAndOrientation(
                terrain, [0, 0, 0.0], [0, 0, 0, 1])
            client.changeDynamics(terrain,
                                               -1,
                                               lateralFriction=1.5)

        if heightfieldSource == useDeepLocoCSV:
            terrainShape = client.createCollisionShape(
                shapeType=client.GEOM_HEIGHTFIELD,
                meshScale=[.5, .5, 5],
                fileName="heightmaps/ground0.txt",
                heightfieldTextureScaling=128)
            terrain = client.createMultiBody(0, terrainShape)
            client.resetBasePositionAndOrientation(
                terrain, [0, 0, 0], [0, 0, 0, 1])
            client.changeDynamics(terrain,
                                               -1,
                                               lateralFriction=1.0)

        if heightfieldSource == useTerrainFromPNG:
            terrainShape = client.createCollisionShape(
                shapeType=client.GEOM_HEIGHTFIELD,
                meshScale=[.5, .5, 80],
                fileName="heightmaps/wm_height_out.png")
            textureId = client.loadTexture(
                "heightmaps/gimp_overlay_out.png")
            terrain = client.createMultiBody(0, terrainShape)
            client.changeVisualShape(terrain,
                                                  -1,
                                                  textureUniqueId=textureId)
            client.resetBasePositionAndOrientation(
                terrain, [0, 0, 5], [0, 0, 0, 10])
            client.changeDynamics(terrain,
                                               -1,
                                               lateralFriction=1.0)

        self.hf_id = terrainShape
        self.terrainShape = terrainShape
        print("TERRAIN SHAPE: {}".format(terrainShape))

        client.changeVisualShape(terrain,
                                              -1,
                                              rgbaColor=[1, 1, 1, 1])

        client.configureDebugVisualizer(
            client.COV_ENABLE_RENDERING, 1)

    def UpdateHeightField(self, heightPerturbationRange=0.08):
        if heightfieldSource == useProgrammatic:
            for j in range(int(numHeightfieldColumns / 2)):
                for i in range(int(numHeightfieldRows / 2)):
                    height = random.uniform(
                        0, heightPerturbationRange)  # +math.sin(time.time())
                    self.heightfieldData[2 * i +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + (2 * j + 1) *
                                         numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 + (2 * j + 1) *
                                         numHeightfieldRows] = height
            #GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of the triangle/heightfield.
            #GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
            #flags = p.GEOM_CONCAVE_INTERNAL_EDGE
            flags = 0
            self.terrainShape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                flags=flags,
                meshScale=[.05, .05, 1],
                heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                heightfieldData=self.heightfieldData,
                numHeightfieldRows=numHeightfieldRows,
                numHeightfieldColumns=numHeightfieldColumns,
                replaceHeightfieldIndex=self.terrainShape)