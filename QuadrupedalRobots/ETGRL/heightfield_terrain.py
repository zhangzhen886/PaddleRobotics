import time
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import pybullet
from scipy.interpolate import interp2d


def unit(x) -> np.ndarray:
    return np.asarray(x) / np.linalg.norm(x)

class Terrain(object):
    def __init__(self, bullet_client):
        self.bullet_client = bullet_client
        self.terrain_id: int = -1
        pass

    @property
    def id(self):
        return self.terrain_id

    def getHeight(self, x, y):
        raise NotImplementedError

    def getNormal(self, x, y):
        raise NotImplementedError

    def getMaxHeightInRange(self, x_range, y_range):
        raise NotImplementedError


class PlainTerrain(Terrain):
    def __init__(self, bullet_client):
        super().__init__(bullet_client)
        self.terrain_id = bullet_client.loadURDF("plane.urdf")
        bullet_client.changeDynamics(self.terrain_id, -1, lateralFriction=5.0)

    def getHeight(self, x, y):
        return 0.0

    def getNormal(self, x, y):
        return np.array((0, 0, 1))

    def getMaxHeightInRange(self, x_range, y_range):
        return 0.0, 0.0, 0.0


class HeightFieldTerrain(Terrain):
    def __init__(self, bullet_client, height_field, resolution, offset=(0., 0., 0.)):
        super().__init__(bullet_client)
        self.height_field = np.asarray(height_field)
        self.offset = np.asarray(offset, dtype=float)
        self.y_dim, self.x_dim = self.height_field.shape
        self.x_size, self.y_size = (self.x_dim - 1) * resolution, (self.y_dim - 1) * resolution
        if isinstance(resolution, Iterable):
            try:
                self.x_rsl, self.y_rsl, self.z_rsl = resolution
            except ValueError:
                self.x_rsl, self.y_rsl = resolution
                self.z_rsl = 1.0
        else:
            self.x_rsl = self.y_rsl = resolution
            self.z_rsl = 1.0
        self.terrain_shape = bullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD, flags=pybullet.GEOM_CONCAVE_INTERNAL_EDGE,
            meshScale=(self.x_rsl, self.y_rsl, self.z_rsl),
            heightfieldTextureScaling=self.x_size,
            heightfieldData=self.height_field.reshape(-1),
            numHeightfieldColumns=self.x_dim, numHeightfieldRows=self.y_dim)

        origin_z = (np.max(self.height_field) + np.min(self.height_field)) / 2
        self.terrain_id = bullet_client.createMultiBody(0, self.terrain_shape)
        bullet_client.changeVisualShape(self.terrain_id, -1, rgbaColor=(1, 1, 1, 1))
        bullet_client.changeDynamics(self.terrain_id, -1, lateralFriction=5.0)
        bullet_client.resetBasePositionAndOrientation(self.terrain_id, self.offset + (0, 0, origin_z), (0, 0, 0, 1))

    @property
    def shape_id(self):
        return self.terrain_shape

    def xCoord2Idx(self, x):
        return int((x + 1e-10 + self.x_size / 2 - self.offset[0]) / self.x_rsl)

    def yCoord2Idx(self, y):
        return int((y + 1e-10 + self.y_size / 2 - self.offset[1]) / self.y_rsl)

    def xIdx2Coord(self, x_idx):
        return x_idx * self.x_rsl - self.x_size / 2 + self.offset[0]

    def yIdx2Coord(self, y_idx):
        return y_idx * self.y_rsl - self.y_size / 2 + self.offset[1]

    @lru_cache(maxsize=20)
    def getNearestVertices(self, x, y):
        x_idx, y_idx = self.xCoord2Idx(x), self.yCoord2Idx(y)
        x_rnd, y_rnd = self.xIdx2Coord(x_idx), self.yIdx2Coord(y_idx)
        if (x - x_rnd) / self.x_rsl + (y - y_rnd) / self.y_rsl < 1:
            v1 = x_rnd, y_rnd, self.height_field[y_idx, x_idx]
        else:
            v1 = x_rnd + self.x_rsl, y_rnd + self.y_rsl, self.height_field[y_idx + 1, x_idx + 1]
        v2 = x_rnd, y_rnd + self.y_rsl, self.height_field[y_idx + 1, x_idx]
        v3 = x_rnd + self.x_rsl, y_rnd, self.height_field[y_idx, x_idx + 1]
        return np.array(v1), np.array(v2), np.array(v3)

    def getMaxHeightInRange(self, x_range, y_range):
        (x_lower, x_upper), (y_lower, y_upper) = x_range, y_range
        x_lower_idx, x_upper_idx = self.xCoord2Idx(x_lower), self.xCoord2Idx(x_upper) + 1
        y_lower_idx, y_upper_idx = self.yCoord2Idx(y_lower), self.yCoord2Idx(y_upper) + 1
        height_field_part = self.height_field[y_lower_idx:y_upper_idx, x_lower_idx:x_upper_idx]
        y_size, x_size = height_field_part.shape
        max_idx = np.argmax(height_field_part)
        max_x_idx, max_y_idx = max_idx % x_size, max_idx // x_size
        max_height = height_field_part[max_y_idx, max_x_idx]
        max_x, max_y = x_lower + max_x_idx * self.x_rsl, y_lower + max_y_idx * self.y_rsl
        return max_x, max_y, max_height

    def getHeight(self, x, y):
        try:
            v1, v2, v3 = self.getNearestVertices(x, y)
        except IndexError:
            return 0.0
        if x == v1[0] and y == v1[1]:
            return v1[2]
        x1, y1, z1 = v2 - v1
        x2, y2, z2 = v3 - v1
        x3, y3 = x - v1[0], y - v1[1]
        div = (x1 * y2 - x2 * y1)
        c1 = (x3 * y2 - x2 * y3) / div
        c2 = (x1 * y3 - x3 * y1) / div
        return c1 * z1 + c2 * z2 + v1[2] + self.offset[2]

    def getNormal(self, x, y) -> np.ndarray:
        try:
            v1, v2, v3 = self.getNearestVertices(x, y)
        except IndexError:
            return np.array((0, 0, 1))
        normal = unit(np.cross(v1 - v2, v1 - v3))
        return normal if normal[2] > 0 else -normal


class RandomUniformTerrain(HeightFieldTerrain):
    def __init__(self,
                 bullet_client,
                 size=15,
                 downsample=20,
                 roughness=0.1,
                 resolution=0.02,
                 offset=(0, 0, 0),
                 seed=None):
        np.random.seed(seed)
        sample_rsl = downsample * resolution
        x = np.arange(-size / 2 - 3 * sample_rsl, size / 2 + 4 * sample_rsl, sample_rsl)
        y = x.copy()
        height_field_downsampled = np.random.uniform(0, roughness, (x.size, y.size))
        self.terrain_func = interp2d(x, y, height_field_downsampled, kind='cubic')

        data_size = int(size / resolution) + 1
        x_upsampled = np.linspace(-size / 2, size / 2, data_size)
        y_upsampled = x_upsampled.copy()
        height_field = self.terrain_func(x_upsampled, y_upsampled)
        super().__init__(bullet_client, height_field, resolution, offset)

    def getHeight(self, x, y):
        return self.terrain_func(x, y).squeeze() + self.offset[2]


class SlopeTerrain(Terrain):
    def __init__(self,
                 bullet_client,
                 size=10,
                 downsample=50,
                 resolution=0.02,
                 offset=(0, 0, 0),
                 ):
        super().__init__(bullet_client)
        self.size, self.offset = size, np.asarray(offset, dtype=float)
        sample_resolution = downsample * resolution
        x = np.arange(-size / 2 - 3 * sample_resolution, size / 2 + 4 * sample_resolution, sample_resolution)
        y = x.copy()
        # height_field_downsampled = (x % 2) / 3
        height_field_downsampled = (np.floor(x / 2) % 2) / 3
        height_field_downsampled = np.repeat(height_field_downsampled, y.shape[0])
        self.terrain_func = interp2d(x, y, height_field_downsampled, kind='linear')  # linear, cubic

        data_size = int(size / resolution) + 1
        x_upsampled = np.linspace(-size / 2, size / 2, data_size)
        y_upsampled = np.linspace(-size / 2, size / 2, data_size)
        self.height_field = self.terrain_func(x_upsampled, y_upsampled)
        terrain_shape = bullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=(resolution, resolution, 1.0),
            heightfieldTextureScaling=size,
            heightfieldData=self.height_field.reshape(-1),
            numHeightfieldColumns=data_size, numHeightfieldRows=data_size)
        z_offset = (np.amax(self.height_field) + np.amin(self.height_field)) / 2
        self.terrain_id = bullet_client.createMultiBody(0, terrain_shape, -1,
                                                        self.offset + (0, 0, z_offset),
                                                        (0, 0, 0, 1))
        bullet_client.changeVisualShape(self.terrain_id, -1, rgbaColor=(1, 1, 1, 1))
        bullet_client.changeDynamics(self.terrain_id, -1, lateralFriction=5.0)

    def getHeight(self, x, y):
        return self.terrain_func(x, y).squeeze() + self.offset[2]

if __name__ == '__main__':
    pybullet.connect(pybullet.GUI)
    t = RandomUniformTerrain(pybullet, size=2)
    # SlopeTerrain(pybullet)
    terrain_visual_shape = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                      radius=0.01,
                                                      rgbaColor=(0., 0.8, 0., 0.6))
    cylinder_shape = pybullet.createVisualShape(shapeType=pybullet.GEOM_CYLINDER,
                                                radius=0.005, length=0.11,
                                                rgbaColor=(0., 0, 0.8, 0.6))

    # from burl.utils.transforms import Quaternion
    # for x in np.linspace(-1, 1, 11):
    #     for y in np.linspace(-1, 1, 11):
    #         h = t.getHeight(x, y)
    #         pybullet.createMultiBody(baseVisualShapeIndex=terrain_visual_shape,
    #                                  basePosition=(x, y, h))
    #         n = t.getNormal(x, y)
    #         y_ax = unit(np.cross(n, (1, 0, 0)))
    #         x_ax = unit(np.cross(y_ax, n))
    #         pybullet.createMultiBody(baseVisualShapeIndex=cylinder_shape,
    #                                  basePosition=(x, y, h),
    #                                  baseOrientation=(Quaternion.from_rotation(np.array((x_ax, y_ax, n)).T)))
    for _ in range(100000):
        pybullet.stepSimulation()
        time.sleep(1 / 240)
