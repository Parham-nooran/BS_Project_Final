import pybullet_data as pd
import random
import cv2


MAZE_NAME = 'test2'
FLAG_TO_FILENAME = {
    'mounts': "heightmaps/wm_height_out.png",
    'maze': f"C:/Users/Parham/Downloads/Project/Maze_Solver/{MAZE_NAME}.png"
    # 'maze': "C:/Users/Parham/Downloads/Project/Maze_Solver/test.png"
}
# C:/Users/Parham/Downloads/Project/Maze_Solver/Maze.png
ROBOT_INIT_POSITION = {
    'mounts': [0, 0, .85],
    'plane': [0, 0, 0.21],
    'hills': [0, 0, 1.98],
    'maze': [0, 0, 0.21],
    'random': [0, 0, 0.21]
}


class Terrain:
    def __init__(self, terrain_source, terrain_id, columns=256, rows=256):
        random.seed(10)
        self.terrain_source = terrain_source
        self.terrain_id = terrain_id
        self.columns = columns
        self.rows = rows

    def generate_terrain(self, client, height_perturbation_range=0.05):
        client.setAdditionalSearchPath(pd.getDataPath())
        client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 0)
        height_perturbation_range = height_perturbation_range
        terrain_data = [0] * self.columns * self.rows
        png_coefficient = 1
        if self.terrain_source == 'random':
            for j in range(int(self.columns / 2)):
                for i in range(int(self.rows / 2)):
                    height = random.uniform(0, height_perturbation_range)
                    terrain_data[2 * i + 2 * j * self.rows] = height
                    terrain_data[2 * i + 1 + 2 * j * self.rows] = height
                    terrain_data[2 * i + (2 * j + 1) * self.rows] = height
                    terrain_data[2 * i + 1 + (2 * j + 1) * self.rows] = height
            terrain_shape = client.createCollisionShape(
                shapeType=client.GEOM_HEIGHTFIELD,
                meshScale=[.5, .5, 1],
                heightfieldTextureScaling=(self.rows - 1) / 2,
                heightfieldData=terrain_data,
                numHeightfieldRows=self.rows,
                numHeightfieldColumns=self.columns)
            terrain = client.createMultiBody(0, terrain_shape)
            client.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

        if self.terrain_source == 'csv':
            terrain_shape = client.createCollisionShape(
                shapeType=client.GEOM_HEIGHTFIELD,
                meshScale=[.3, .3, .3],
                fileName="heightmaps/ground0.txt",
                heightfieldTextureScaling=128)
            terrain = client.createMultiBody(0, terrain_shape)
            client.changeVisualShape(terrain, -1)
            client.resetBasePositionAndOrientation(terrain, [1, 0, 2], [0, 0, 0, 1])

        if self.terrain_source == 'png':
            img = cv2.imread(FLAG_TO_FILENAME[self.terrain_id])
            x, y, _ = img.shape
            terrain_shape = client.createCollisionShape(
                shapeType=client.GEOM_HEIGHTFIELD,
                meshScale=[png_coefficient, png_coefficient, 2.8],
                fileName=FLAG_TO_FILENAME[self.terrain_id])
            terrain = client.createMultiBody(0, terrain_shape)
            if self.terrain_id == "mounts":
                textureId = client.loadTexture("heightmaps/gimp_overlay_out.png")
                client.changeVisualShape(terrain, -1, textureUniqueId=textureId)
            client.resetBasePositionAndOrientation(terrain, [-y/2, x/2, 1], [0, 0, 0, 1])
            # client.resetBasePositionAndOrientation(terrain, [-10, -10, 1], [0, 0, 0, 1])

        self.terrain_shape = terrain_shape
        client.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
