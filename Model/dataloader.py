import numpy as np
import random
import pickle
# import math
from PIL import Image
import json
# import scipy.ndimage
import imageio.v2 as imageio
import math

# import cv2


image_size = 256
vector_norm = 25.0


# rotate
# (1) sat_img
# (2) gt_seg
# (3) neighbors
# (4) sample point
# (5) sample mask?


# angle is in degrees
def rotate(sat_img, gt_seg, neighbors, samplepoints, angle=0, size=2048):
    mask = np.zeros(np.shape(gt_seg))

    mask[256: size - 256, 256: size - 256] = 1

    # rotate sat_img
    if sat_img.dtype != np.uint8:
        if sat_img.max() <= 1.0:  # assume the image data range is 0-1
            sat_img = (sat_img * 255).astype(np.uint8)
        else:  # assume the image data range is 0-255
            sat_img = sat_img.astype(np.uint8)

    sat_img = np.array(
        Image.fromarray(sat_img).rotate(angle, resample=Image.BICUBIC, expand=False)
        # omit the error code here,already fixed
    )

    # 旋转 gt_seg
    if gt_seg.dtype != np.uint8:
        gt_seg = gt_seg.astype(np.uint8)

    gt_seg = np.array(
        Image.fromarray(gt_seg).rotate(angle, resample=Image.BICUBIC, expand=False)
    )

    # make sure mask is in uint8
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    mask = np.array(
        Image.fromarray(mask).rotate(angle, resample=Image.BICUBIC, expand=False)
    )

    new_neighbors = {}
    new_samplepoints = {}

    def transfer(pos, angle):
        x = pos[0] - int(size / 2)
        y = pos[1] - int(size / 2)

        new_x = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
        new_y = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))

        return int(new_x + int(size / 2)), int(new_y + int(size / 2))

    def inrange(pos, m):
        if (
                m < pos[0] < size - 1 - m
                and m < pos[1] < size - 1 - m
        ):
            return True
        else:
            return False

    for k, n in neighbors.items():
        nk = transfer(k, angle)

        if not inrange(nk, 0):
            continue

        new_neighbors[nk] = []

        for nei in n:
            nn = transfer(nei, angle)
            if inrange(nn, 0):
                new_neighbors[nk].append(nn)

    for k, vs in samplepoints.items():

        new_samplepoints[k] = []

        for v in vs:
            nv = transfer(v, angle)

            if inrange(nv, 256):
                new_samplepoints[k].append(nv)
    # print(f"shape of sat_img when rotate called = {sat_img.shape}") (2048, 2048, 3)
    # print(f"shape of gt_seg when rotate called = {gt_seg.shape}")  (2048, 2048)

    return sat_img, gt_seg, new_neighbors, new_samplepoints, mask


def neighbor_transpos(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (k[1], k[0])
        nv = []

        for _v in v:
            nv.append((_v[1], _v[0]))

        n_out[nk] = nv

    return n_out


def neighbor_to_integer(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))

        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []

        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))

            if new_n_k in nv:
                pass
            else:
                nv.append(new_n_k)

        n_out[nk] = nv

    return n_out


class Sat2GraphDataLoader:
    def __init__(
            self,
            folder,
            indrange=[0, 10],
            imgsize=256,
            preload_tiles=4,
            max_degree=6,
            loadseg=False,
            random_mask=True,
            testing=False,
            dataset_image_size=2048,
            transpose=False,
    ):
        self.folder = folder
        self.indrange = indrange
        self.random_mask = random_mask
        self.dataset_image_size = dataset_image_size
        self.transpose = transpose

        self.preload_tiles = preload_tiles
        self.max_degree = max_degree
        self.num = 0
        self.loadseg = loadseg

        self.image_size = imgsize
        self.testing = testing
        global image_size
        image_size = imgsize

        self.input_sat = np.zeros((8, image_size, image_size, 3))
        # print(f"shape of input_sat when dataloader called = {self.input_sat.shape}")
        # shape of input_sat when dataloader called = (8, 352, 352, 3)

        self.gt_seg = np.zeros((8, 1, image_size, image_size))
        self.target_prob = np.zeros((8, 2 * (max_degree + 1), image_size, image_size))
        self.target_vector = np.zeros((8, 2 * max_degree, image_size, image_size))
        # print(f"shape of target_vector when dataloader called = {self.target_vector.shape}")
        # shape of target_vector when dataloader called = (8, 352, 352, 12)

        self.noise_mask = (np.random.rand(3, 64, 64) - 0.5) * 0.8

        random.seed(1)

    def loadtile(self, ind):
        try:
            sat_img = imageio.imread(self.folder + "/region_%d_sat.png" % ind).astype(float)
        except:
            sat_img = imageio.imread(self.folder + "/region_%d_sat.jpg" % ind).astype(float)
        print(f"shape of sat_img when loadtile called = {sat_img.shape}")
        max_v = np.amax(sat_img) + 0.0001

        sat_img = (sat_img.astype(float) / max_v - 0.5) * 0.9

        # 1,2048,2048,3
        sat_img = sat_img.reshape(
            (1, 3, self.dataset_image_size, self.dataset_image_size)
        )
        print(f"shape of sat_img when loadtile after reshape = {sat_img.shape}")

        # Image.fromarray(((sat_img[0,:,:,:] + 0.5) * 255.0).astype(np.uint8)).save("outputs/test.png")

        # print(np.shape(sat_img))

        # 1,352,352,14 → 1, 14, 352, 352
        tiles_prob = np.zeros(
            (
                1,
                2 * (self.max_degree + 1),
                self.dataset_image_size,
                self.dataset_image_size,

            )
        )
        # 1,352,352,12 → 1, 12, 352, 352
        tiles_vector = np.zeros(
            (1, 2 * self.max_degree, self.dataset_image_size, self.dataset_image_size)
        )

        tiles_prob[:, 0::2, :, :] = 0
        tiles_vector[:, 1::2, :, :] = 1

        try:
            with open(self.folder + "/region_%d_refine_gt_graph.p" % ind, 'rb') as file:
                neighbors = pickle.load(file)
            neighbors = neighbor_to_integer(neighbors)

            if self.transpose:
                neighbors = neighbor_transpos(neighbors)

            r = 1
            i = 0

            # tiles_angle = np.zeros((self.dataset_image_size, self.dataset_image_size, 1), dtype=np.uint8)

            for loc, n_locs in neighbors.items():
                if (
                        loc[0] < 16
                        or loc[1] < 16
                        or loc[0] > self.dataset_image_size - 16
                        or loc[1] > self.dataset_image_size - 16
                ):
                    continue

                tiles_prob[i, 0, loc[0], loc[1]] = 1
                tiles_prob[i, 1, loc[0], loc[1]] = 0

                for x in range(loc[0] - r, loc[0] + r + 1):
                    for y in range(loc[1] - r, loc[1] + r + 1):
                        tiles_prob[i, 0, x, y] = 1
                        tiles_prob[i, 1, x, y] = 0

                for n_loc in n_locs:
                    if (
                            n_loc[0] < 16
                            or n_loc[1] < 16
                            or n_loc[0] > self.dataset_image_size - 16
                            or n_loc[1] > self.dataset_image_size - 16
                    ):
                        continue
                    d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi

                    j = int(d / (math.pi / 3.0)) % self.max_degree

                    for x in range(loc[0] - r, loc[0] + r + 1):
                        for y in range(loc[1] - r, loc[1] + r + 1):
                            tiles_prob[i, 2 + 2 * j, x, y] = 1
                            tiles_prob[i, 2 + 2 * j + 1, x, y] = 0

                            tiles_vector[i, 2 * j, x, y] = (
                                                                   n_loc[0] - loc[0]
                                                           ) / vector_norm
                            tiles_vector[i, 2 * j + 1, x, y] = (
                                                                       n_loc[1] - loc[1]
                                                               ) / vector_norm
        except:
            pass

        print(f"shape of sat_img when loadtile return = {sat_img.shape}")
        print(f"shape of tiles_prob when loadtile return = {tiles_prob.shape}")
        print(f"shape of tiles_vector when loadtile return = {tiles_vector.shape}")
        return sat_img, tiles_prob, tiles_vector

    # num = 1024 is useless here.
    """Batch_size, channel, image_size, image_size"""

    # read tile?
    def preload(self, num=1024, seg_only=False):
        self.noise_mask = (np.random.rand(64, 64, 3)) * 1.0 + 0.5

        image_size = self.image_size

        tiles = []

        """ 初始化全零的 NumPy 数组，对数据进行预加载。"""
        # 初始化为一个形状为 (4, 2048, 2048, 3) 的全零数组。表示有 4 个预加载的图像块，每个图像块大小为 2048x2048，且有 3 个通道（例如 RGB 图像）。
        # initialize the array with zeros, shape is (4, 2048, 2048, 3).
        self.tiles_input = np.zeros(
            (self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 3)
        )

        # 初始化为一个形状为 (4, 1, 2048, 2048) 的全零数组。表示有 4 个预加载的图像块，每个图像块大小为 2048x2048，且有 1 个通道，用于存储地面真值分割数据。
        # initialize the array with zeros, shape is (4, 1, 2048, 2048) for ground truth segmentation data.
        self.tiles_gt_seg = np.zeros(
            (self.preload_tiles, 1, self.dataset_image_size, self.dataset_image_size)
        )

        # 初始化为一个形状为 (4, 14, 2048, 2048) 的全零数组。表示有 4 个预加载的图像块，每个图像块大小为 2048x2048，且有 14 个通道，用于存储概率图。
        # initialize the array with zeros, shape is (4, 14, 2048, 2048) for probability map.
        self.tiles_prob = np.zeros(
            (
                self.preload_tiles,
                2 * (self.max_degree + 1),
                self.dataset_image_size,
                self.dataset_image_size,

            )
        )
        # 初始化为一个形状为  (4, 12, 2048, 2048) 的数组。表示有 4 个预加载的图像块，每个图像块大小为 2048x2048，且有 12 个通道（2*6）。这些通道用于存储与旋转相关的目标向量。
        # initialize the array with zeros, shape is (4, 12, 2048, 2048) for target vector related to rotation.
        self.tiles_vector = np.zeros(
            (
                self.preload_tiles,
                2 * self.max_degree,
                self.dataset_image_size,
                self.dataset_image_size,

            )
        )

        self.tiles_prob[:, 0::2, :, :] = 0  # all even channels of the array
        self.tiles_prob[:, 1::2, :, :] = 1  # all odd channels of the array

        # 初始化为一个形状为 (4, 2048, 2048) 的全 1 数组。表示 4 个预加载的图像块中所有的像素点都满足 rotmask 的条件。
        # initialize the array with ones, shape is (4, 2048, 2048).
        # represents all pixels in 4 preloaded image blocks satisfy the condition of rotmask.
        self.rotmask = np.ones(
            (self.preload_tiles, self.dataset_image_size, self.dataset_image_size)
        )

        self.samplepoints = []

        """函数体"""
        for i in range(self.preload_tiles):
            ind = random.choice(self.indrange)

            # load sample points
            samplepoints = json.load(
                open(
                    self.folder + "/region_%d_refine_gt_graph_samplepoints.json" % ind,
                    "r",
                )
            )
            self.samplepoints.append(samplepoints)
            # samplepoints loaded successfully while preloading

            try:
                sat_img = imageio.imread(
                    self.folder + "/region_%d_sat.png" % ind
                ).astype(float)
            except:
                sat_img = imageio.imread(
                    self.folder + "/region_%d_sat.jpg" % ind
                ).astype(float)
            # sat_img = sat_img.transpose((2, 0, 1))
            # print(f"shape of sat_img when preload called = {sat_img.shape}")  (2048, 2048, 3)
            max_v = np.amax(sat_img) + 0.0001

            with open(self.folder + "/region_%d_refine_gt_graph.p" % ind, 'rb') as file:
                neighbors = pickle.load(file)

            neighbors = neighbor_to_integer(neighbors)
            # neighbors loaded successfully from refine_gt_graph.p while preloading

            if self.transpose:
                neighbors = neighbor_transpos(neighbors)

            gt_seg = imageio.imread(self.folder + "/region_%d_gt.png" % ind)

            self.rotmask[i, :, :] = np.ones(
                (self.dataset_image_size, self.dataset_image_size)
            )

            # random rotation augmentation
            # rotate
            # (1) sat_img
            # (2) gt_seg
            # (3) neighbors
            # (4) sample point
            # (5) sample mask?

            if self.testing == False and random.randint(0, 5) < 4:
                angle = random.randint(0, 3) * 90 + random.randint(-30, 30)
                sat_img, gt_seg, neighbors, samplepoints, rotmask = rotate(
                    sat_img,
                    gt_seg,
                    neighbors,
                    samplepoints,
                    angle=angle,
                    size=self.dataset_image_size,
                )
                self.rotmask[i, :, :] = rotmask
                # print(f"shape of sat_img after rotate = {sat_img.shape}") (2048, 2048, 3)
                # print(f"shape of gt_seg after rotate = {gt_seg.shape}")

            # print(f"shape of tile_input = {self.tiles_input.shape}") (4, 2048, 2048, 3)
            # print(f"shape of sat_img after rotate = {sat_img.shape}") (2048, 2048, 3)

            # 将旋转后的卫星图像归一化处理后，存储在 self.tiles_input 的第 i 个图像块中。归一化过程将图像像素值缩放到 [-0.5, 0.5] 范围内。
            # Process the rotated satellite image and store it in the i-th image block of self.tiles_input.
            # The normalization process scales the pixel values of the image to the range [-0.5, 0.5].
            self.tiles_input[i, :, :, :] = sat_img.astype(float) / max_v - 0.5

            # 将地面真值分割图像也进行归一化处理，并存储在 self.tiles_gt_seg 的第 i 个图像块中，范围为 [-0.5, 0.5]。
            # Normalize the ground truth segmentation image and store it in the i-th image block of self.tiles_gt_seg, range is [-0.5, 0.5].
            self.tiles_gt_seg[i, 0, :, :] = gt_seg.astype(float) / 255.0 - 0.5

            # Define a radius r = 1 for subsequent neighborhood processing.
            r = 1

            for loc, n_locs in neighbors.items():
                # 检查 loc 是否位于图像的有效区域内。如果 loc 坐标在边缘区域（距离图像边缘小于 16 像素），则跳过该像素，继续处理下一个像素。
                # Check if loc is in the valid area of the image. If the loc coordinate is in the edge area
                # (less than 16 pixels away from the edge of the image), skip the pixel and continue to process the next pixel.
                if (
                        loc[0] < 16
                        or loc[1] < 16
                        or loc[0] > self.dataset_image_size - 16
                        or loc[1] > self.dataset_image_size - 16
                ):
                    continue

                # 在 self.tiles_prob 的第 i 个图像块中，设置 loc 对应位置的概率值。通道 0 设置为 1，通道 1 设置为 0。这表示该位置的某个二分类状态。
                # Set the probability value of the location corresponding to loc in the i-th image block of self.tiles_prob.
                self.tiles_prob[i, 0, loc[0], loc[1]] = 1
                self.tiles_prob[i, 1, loc[0], loc[1]] = 0

                # 对于 loc 周围半径为 1 的区域（3x3 的范围），设置相同的概率值，表示在这个邻域内所有像素都属于相同的类别。

                for x in range(loc[0] - r, loc[0] + r + 1):
                    for y in range(loc[1] - r, loc[1] + r + 1):
                        self.tiles_prob[i, 0, x, y] = 1
                        self.tiles_prob[i, 1, x, y] = 0

                """
                对于每个邻居位置 n_loc，首先检查其是否位于图像的有效区域内（避免边缘像素）。
                计算 n_loc 和 loc 之间的角度 d，并将其归一化为 0 到 2π 之间。计算 j，它对应在概率图中的一个方向通道。
                对于 loc 周围半径为 1 的区域，设置方向相关的概率值和向量值。self.tiles_prob 对应的 2 + 2 * j 和 2 + 2 * j + 1 通道表示该方向的概率，self.tiles_vector 的相应通道存储归一化的向量方向。
                """
                # For each neighboring position n_loc, first check if it is in the valid area of the image (avoid
                # edge pixels). Calculate the angle d between n_loc and loc, and normalize it to be between 0 and 2π.
                # Calculate j, which corresponds to a directional channel in the probability map.

                # For the area around loc with a radius of 1, set the probability value and vector value related to
                # the direction. The 2 + 2 * j and 2 + 2 * j + 1 channels of self.tiles_prob correspond to the
                # probability of that direction, and the corresponding channel of self.tiles_vector stores the
                # normalized vector direction.

                for n_loc in n_locs:

                    if (
                            n_loc[0] < 16
                            or n_loc[1] < 16
                            or n_loc[0] > self.dataset_image_size - 16
                            or n_loc[1] > self.dataset_image_size - 16
                    ):
                        continue

                    d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi
                    j = int(d / (math.pi / 3.0)) % self.max_degree

                    for x in range(loc[0] - r, loc[0] + r + 1):
                        for y in range(loc[1] - r, loc[1] + r + 1):
                            self.tiles_prob[i, 2 + 2 * j, x, y] = 1
                            self.tiles_prob[i, 2 + 2 * j + 1, x, y] = 0

                            self.tiles_vector[i, 2 * j, x, y] = (
                                                                        n_loc[0] - loc[0]
                                                                ) / vector_norm
                            self.tiles_vector[i, 2 * j + 1, x, y] = (
                                                                            n_loc[1] - loc[1]
                                                                    ) / vector_norm

            """
            如果当前不是测试模式（self.testing == False），对输入图像块 tiles_input 进行颜色抖动和归一化调整，以增强数据的多样性：
            首先，对整个图像应用一个随机的缩放和偏移操作，然后将结果限制在 [-0.5, 0.5] 的范围内。
            然后，对每个通道（红、绿、蓝）分别应用随机的缩放调整。
            """
            # If it is not in testing mode (self.testing == False), color jitter and normalization adjustments are
            # applied to the input image block tiles_input to enhance the diversity of the data:
            # First, apply a random scaling and offset operation to the entire image, and then limit the result to the
            # range of [-0.5, 0.5]. Then, apply random scaling adjustments to each channel (red, green, blue) separately.

            if not self.testing:
                # print(f"shape of tiles_input when preload called = {self.tiles_input.shape}"), (4, 2048, 2048, 3)
                self.tiles_input[i, :, :, :] = self.tiles_input[i, :, :, :] * (
                        0.8 + 0.2 * random.random()
                ) - (random.random() * 0.4 - 0.2)
                # 限制在 ±0.5 之间
                # Limit to between ±0.5
                self.tiles_input[i, :, :, :] = np.clip(
                    self.tiles_input[i, :, :, :], -0.5, 0.5
                )

                # 对每个通道（红、绿、蓝）分别应用随机的缩放调整。
                # Apply random scaling adjustments to each channel (red, green, blue) separately.
                self.tiles_input[i, 0, :, :] = self.tiles_input[i, 0, :, :] * (
                        0.8 + 0.2 * random.random()
                )
                self.tiles_input[i, 1, :, :] = self.tiles_input[i, 1, :, :] * (
                        0.8 + 0.2 * random.random()
                )
                self.tiles_input[i, 2, :, :] = self.tiles_input[i, 2, :, :] * (
                        0.8 + 0.2 * random.random()
                )
                # print(f"shape of tiles_input when preload called = {self.tiles_input.shape}") (4, 2048, 2048, 3)

    def getBatch(self, batchsize=64, st=None):

        image_size = self.image_size
        # print(f"shape of image_size when getbatch called = {image_size}") 352
        # print(f"shape of self.image_size when getbatch called = {self.image_size}") 352
        # print(f"shape of self.dataset_image_size when getbatch called = {self.dataset_image_size}") 2048

        for i in range(batchsize):
            c = 0
            while True:
                # tile_id：从预加载的图像块中随机选择一个图像块（tile_id）。self.preload_tiles 表示预加载的图像块数量（例如，4 个），tile_id 的值将在 0 到
                # self.preload_tiles - 1 之间随机选择。
                # tile_id: randomly select an image block (tile_id) from the preloaded image blocks. self.preload_tiles
                # represents the number of preloaded image blocks (e.g., 4), and the value of tile_id will be randomly
                # selected between 0 and self.preload_tiles - 1.
                tile_id = random.randint(0, self.preload_tiles - 1)

                # coin：生成一个 0 到 99 之间的随机数，用于决定接下来如何从选定的图像块中提取图像区域。 coin: generate a random number between 0 and 99 to
                # determine how to extract the image area from the selected image block.
                coin = random.randint(0, 99)

                if coin < 20:  # 20%
                    while True:
                        # 随机选择坐标：从图像块中随机选择一个 (x, y) 坐标，确保该坐标在图像的有效范围内（远离图像边缘 256 像素）。 有效范围：这里的范围是从 256 到
                        # dataset_image_size - 256 - image_size，确保采样区域不会越界。

                        # Randomly select coordinates: randomly select an (x, y) coordinate from the image block,
                        # ensuring that the coordinate is within the valid range of the image (far from the image
                        # edge by 256 pixels). Valid range: the range here is from 256 to dataset_image_size - 256 -
                        # image_size, ensuring that the sampling area will not go out of bounds.

                        x = random.randint(
                            256, self.dataset_image_size - 256 - image_size
                        )
                        y = random.randint(
                            256, self.dataset_image_size - 256 - image_size
                        )

                        # 检查掩码：如果在随机选择的 (x, y) 坐标处，self.rotmask 的值大于 0.5（表示该位置有效），则退出内层循环，继续生成样本。
                        if self.rotmask[tile_id, x, y] > 0.5:
                            break

                elif coin < 40:  # complicated intersections
                    sps = self.samplepoints[tile_id]["complicated_intersections"]

                    if len(sps) == 0:
                        c += 1
                        continue

                    ind = random.randint(0, len(sps) - 1)

                    x = sps[ind][0] - image_size / 2
                    y = sps[ind][1] - image_size / 2

                    x = np.clip(x, 256, self.dataset_image_size - 256 - image_size)
                    y = np.clip(y, 256, self.dataset_image_size - 256 - image_size)

                elif coin < 60:  # parallel roads
                    sps = self.samplepoints[tile_id]["parallel_road"]

                    if len(sps) == 0:
                        c += 1
                        continue

                    ind = random.randint(0, len(sps) - 1)

                    x = sps[ind][0] - image_size / 2
                    y = sps[ind][1] - image_size / 2

                    x = np.clip(x, 256, self.dataset_image_size - 256 - image_size)
                    y = np.clip(y, 256, self.dataset_image_size - 256 - image_size)

                else:  # overpass
                    sps = self.samplepoints[tile_id]["overpass"]

                    if len(sps) == 0:
                        c += 1
                        continue

                    ind = random.randint(0, len(sps) - 1)

                    x = sps[ind][0] - image_size / 2
                    y = sps[ind][1] - image_size / 2

                    x = np.clip(x, 256, self.dataset_image_size - 256 - image_size)
                    y = np.clip(y, 256, self.dataset_image_size - 256 - image_size)

                x = int(x)
                y = int(y)
                # print("x = ", x) x =  1440
                # print("y = ", y) y =  1275

                c += 1
                # print("np.sum", np.sum(self.tiles_gt_seg[tile_id, :, x: x + image_size, y: y + image_size] ))

                if (
                        np.sum(
                            self.tiles_gt_seg[
                            tile_id, :, x: x + image_size, y: y + image_size
                            ]
                            + 0.5
                        )
                        < 20 * 20
                        and c < 10
                ):
                    continue
                # shape?
                # print(f"shape of input_sat when getbatch called = {self.input_sat.shape}")
                # # shape of input_sat when getbatch called = (8, 352, 352, 3)
                # print(f"shape of tiles_input when getbatch called = {self.tiles_input.shape}")
                # # shape of tiles_input when getbatch called = (9, 2048, 2048, 3)
                # print(f"tile_id when getbatch called = {tile_id}")  # 1
                # print(image_size)  # 352
                self.input_sat[i, :, :, :] = self.tiles_input[
                                             tile_id, x: x + image_size, y: y + image_size, :
                                             ]

                if random.randint(0, 100) < 50 and self.random_mask == True:

                    # add noise
                    for it in range(random.randint(1, 5)):
                        xx = random.randint(0, image_size - 64 - 1)
                        yy = random.randint(0, image_size - 64 - 1)

                        self.input_sat[i, xx: xx + 64, yy: yy + 64, :] = (
                                np.multiply(
                                    self.input_sat[i, xx: xx + 64, yy: yy + 64, :] + 0.5,
                                    self.noise_mask,
                                )
                                - 0.5
                        )

                    # add more noise
                    for it in range(random.randint(1, 3)):
                        xx = random.randint(0, image_size - 64 - 1)
                        yy = random.randint(0, image_size - 64 - 1)

                        self.input_sat[i, xx: xx + 64, yy: yy + 64, :] = (
                                self.noise_mask - 1.0
                        )

                self.target_prob[i, :, :, :] = self.tiles_prob[
                                               tile_id, :, x: x + image_size, y: y + image_size
                                               ]
                # print(f"shape of target_prob when getbatch called = {self.target_prob.shape}")
                self.target_vector[i, :, :, :] = self.tiles_vector[
                                                 tile_id, :, x: x + image_size, y: y + image_size
                                                 ]
                # print(f"shape of target_vector when getbatch called = {self.target_vector.shape}")
                self.gt_seg[i, :, :, :] = self.tiles_gt_seg[
                                          tile_id, :, x: x + image_size, y: y + image_size
                                          ]
                # print(f"shape of gt_seg when getbatch called = {self.gt_seg.shape}")

                """
                shape of target_prob when getbatch called = (8, 352, 352, 14)
                shape of target_vector when getbatch called = (8, 352, 352, 12)
                shape of gt_seg when getbatch called = (8, 352, 352, 1)
                """
                break

        st = 0
        # print(f"shape of input_sat getbatch return: = {self.input_sat[st : st + batchsize, :, :, :].shape}")
        # print(f"shape of target_prob getbatch return: = {self.target_prob[st : st + batchsize, :, :, :].shape}")
        # print(f"shape of target_vector getbatch return: = {self.target_vector[st : st + batchsize, :, :, :].shape}")
        # print(f"shape of gt_seg getbatch return: = {self.gt_seg[st : st + batchsize, :, :, :].shape}")

        """
        so far so good
        
        shape of input_sat getbatch return: = (2, 352, 352, 3)
        shape of target_prob getbatch return: = (2, 352, 352, 14)
        shape of target_vector getbatch return: = (2, 352, 352, 12)
        shape of gt_seg getbatch return: = (2, 352, 352, 1)
        """
        # 示例: 对于一个 256x256 的二分类任务的地面真值分割图像，gt_seg 的形状为 (1, 1, 256, 256)，其中每个像素的值是 0 或 1，表示该像素属于背景或前景。

        # for a binary ground truth segmentation image of a 256x256 binary task, the shape of gt_seg is (1, 1, 256,
        # 256), where the value of each pixel is 0 or 1, indicating that the pixel belongs to the background or
        # foreground.
        return (
            self.input_sat[st: st + batchsize, :, :, :],
            self.target_prob[st: st + batchsize, :, :, :],
            self.target_vector[st: st + batchsize, :, :, :],
            self.gt_seg[st: st + batchsize, :, :, :],
        )
