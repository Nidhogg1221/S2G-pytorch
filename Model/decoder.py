import cv2
import imageio.v2 as imageio
from PIL import Image
from scipy.ndimage import gaussian_filter, minimum_filter
import scipy.ndimage.morphology as morphology
from rtree import index
import pickle
from common import *
import torch

vector_norm = 25.0

def vNorm(v1):
    if isinstance(v1, torch.Tensor):
        v1 = v1.cpu().numpy()

    l = distance(v1, (0, 0)) + 0.0000001
    return (v1[0] / l, v1[1] / l)


def anglediff(v1, v2):
    v1 = vNorm(v1)
    v2 = vNorm(v2)

    return v1[0] * v2[0] + v1[1] * v2[1]


# remove isolate nodes and short edges

# threshold
def graph_refine(graph, isolated_thr=150, spurs_thr=30, three_edge_loop_thr=70):
    """
    graph: adjacency list, 邻接表，{node1,: [neighbor1, neighbor2, ...], node2, []...}
    key → node: (x, y)
    value → neighbor list [neighbor1, neighbor2, ...]
    """
    neighbors = graph

    gid = 0  # groupID
    """
    dict,
    key → group id
    value → (node_counter, edge_length_sum)
    """
    grouping = {}

    for k, v in neighbors.items():
        if k not in grouping:
            # start a search
            """
            Apply bfs
            Divide nodes into groups
            """
            queue = [k]

            while len(queue) > 0:
                n = queue.pop(0)

                if n not in grouping:
                    grouping[n] = gid
                    for nei in neighbors[n]:
                        queue.append(nei)

            gid += 1

    # dict, {node k, group id v}
    group_count = {}

    # calculate
    for k, v in grouping.items():
        if v not in group_count:
            group_count[v] = (1, 0)  # 1 node, 0 edge
        else:
            # update num of nodes
            group_count[v] = (group_count[v][0] + 1, group_count[v][1])

        # sum
        for nei in neighbors[k]:  # k:  node coordinate
            a = k[0] - nei[0]  # x direction
            b = k[1] - nei[1]  # y direction

            d = np.sqrt(a * a + b * b)  # euclidean distance

            group_count[v] = (group_count[v][0], group_count[v][1] + d // 2)  # since each edge will be counted twice

    # short spurs 支线 지선
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:  # only has 1 neighbor node
            if len(neighbors[v[0]]) >= 3:  # unless the only neighbor node has more than 3 neighbors, skip
                # calculate the distance
                a = k[0] - v[0][0]
                b = k[1] - v[0][1]

                d = np.sqrt(a * a + b * b)

                if d < spurs_thr:  # check the distance
                    remove_list.append(k)

    remove_list2 = []
    remove_counter = 0
    new_neighbors = {}

    # 4 cases
    def isRemoved(k):
        gid = grouping[k]
        if group_count[gid][0] <= 1:
            return True
        elif group_count[gid][1] <= isolated_thr:
            return True
        elif k in remove_list:
            return True
        elif k in remove_list2:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    #print(len(new_neighbors), "remove", remove_counter, "nodes")

    return new_neighbors


# remove short edges
def graph_shave(graph, spurs_thr=50):
    neighbors = graph

    # short spurs
    remove_list = []
    # short edge detect
    for k, v in neighbors.items():
        # end node check
        if len(v) == 1:
            d = distance(k, v[0])  # distance of current node and neighbor node
            cur = v[0]  # next node
            l = [k]
            while True:
                # when length of current node >= 3
                if len(neighbors[cur]) >= 3:
                    break
                elif len(neighbors[cur]) == 1:
                    l.append(cur)
                    break

                else:

                    if neighbors[cur][0] == l[-1]:
                        next_node = neighbors[cur][1]  # already visited
                    else:
                        next_node = neighbors[cur][0]

                    d += distance(cur, next_node)
                    l.append(cur)

                    cur = next_node

            if d < spurs_thr:
                for n in l:
                    if n not in remove_list:
                        remove_list.append(n)

    def isRemoved(k):
        if k in remove_list:
            return True
        else:
            return False

    new_neighbors = {}
    remove_counter = 0

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    #print("shave", len(new_neighbors), "remove", remove_counter, "nodes")

    return new_neighbors


def graph_refine_deloop(neighbors, max_step=10, max_length=200, max_diff=5):
    removed = []
    impact = []

    remove_edge = []
    new_edge = []

    for k, v in neighbors.items():
        if k in removed:
            continue

        if k in impact:
            continue

        if len(v) < 2:
            continue

        # traversal neighbor nodes
        for nei1 in v:
            if nei1 in impact:
                continue

            if k in impact:
                continue

            for nei2 in v:
                if nei2 in impact:
                    continue
                if nei1 == nei2:
                    continue

                # edge loop detection, check the similarity of cos
                # defined in common.py
                if neighbors_cos(neighbors, k, nei1, nei2) > 0.984:  # nearly 10°
                    l1 = neighbors_dist(neighbors, k, nei1)
                    l2 = neighbors_dist(neighbors, k, nei2)

                    #print("candidate!", l1,l2,neighbors_cos(neighbors, k, nei1, nei2))

                    if l2 < l1:
                        nei1, nei2 = nei2, nei1  # make nei2 has larger distance

                    remove_edge.append((k, nei2))
                    remove_edge.append((nei2, k))

                    new_edge.append((nei1, nei2))

                    impact.append(k)
                    impact.append(nei1)
                    impact.append(nei2)

                    break

    new_neighbors = {}

    def isRemoved(k):
        if k in removed:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                elif (nei, k) in remove_edge:
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    for new_e in new_edge:
        nk1 = new_e[0]
        nk2 = new_e[1]

        if nk2 not in new_neighbors[nk1]:
            new_neighbors[nk1].append(nk2)
        if nk1 not in new_neighbors[nk2]:
            new_neighbors[nk2].append(nk1)

    #print("remove %d edges" % len(remove_edge))

    return new_neighbors, len(remove_edge)


# Apply r tree index to find the stacking road
def locate_stacking_road(graph):
    # r-tree index?
    idx = index.Index()

    edges = []

    for n1, v in graph.items():
        for n2 in v:
            if (n1, n2) in edges or (n2, n1) in edges:
                continue

            # 计算边界
            x1 = min(n1[0], n2[0])
            x2 = max(n1[0], n2[0])

            y1 = min(n1[1], n2[1])
            y2 = max(n1[1], n2[1])

            idx.insert(len(edges), (x1, y1, x2, y2))

            edges.append((n1, n2))

    adjustment = {}

    crossing_point = {}

    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]

        x1 = min(n1[0], n2[0])
        x2 = max(n1[0], n2[0])

        y1 = min(n1[1], n2[1])
        y2 = max(n1[1], n2[1])

        candidates = list(idx.intersection((x1, y1, x2, y2)))

        for _candidate in candidates:
            # todo mark the overlap point
            candidate = edges[_candidate]

            # 跳过共用顶点的边
            if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
                continue
            # 相交
            if intersect(n1, n2, candidate[0], candidate[1]):
                ip = intersectPoint(n1, n2, candidate[0], candidate[1])
                # 如果没有记录交点，则记录
                if (candidate, edge) not in crossing_point:
                    crossing_point[(edge, candidate)] = ip

                #release points

                d = distance(ip, n1)
                thr = 5.0
                if d < thr:
                    vec = neighbors_norm(graph, n1, n2)  # 归一化向量
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))

                    if n1 not in adjustment:
                        adjustment[n1] = [vec]
                    else:
                        adjustment[n1].append(vec)

                d = distance(ip, n2)
                if d < thr:
                    vec = neighbors_norm(graph, n2, n1)
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))

                    if n2 not in adjustment:
                        adjustment[n2] = [vec]
                    else:
                        adjustment[n2].append(vec)

                # candidate vertex 进行调整
                c1 = candidate[0]
                c2 = candidate[1]

                d = distance(ip, c1)
                if d < thr:
                    vec = neighbors_norm(graph, c1, c2)
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))

                    if c1 not in adjustment:
                        adjustment[c1] = [vec]
                    else:
                        adjustment[c1].append(vec)

                d = distance(ip, c2)
                if d < thr:
                    vec = neighbors_norm(graph, c2, c1)
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))

                    if c2 not in adjustment:
                        adjustment[c2] = [vec]
                    else:
                        adjustment[c2].append(vec)

    return crossing_point, adjustment


def _vis(_node_neighbors, save_file, size=2048, bk=None, draw_intersection=False):
    node_neighbors = _node_neighbors

    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    color_node = (255, 0, 0)

    if bk is not None:
        img = imageio.imread(bk)

        img = img.astype(float)
        img = (img - 127) * 0.75 + 127
        img = img.astype(np.uint8)

        color_edge = (0, 255, 255)  # yellow
    else:
        color_edge = (0, 0, 0)  # black

    edge_width = 2

    # draw edge
    for k, v in node_neighbors.items():
        n1 = k
        for n2 in v:
            cv2.line(img, (n1[1], n1[0]), (n2[1], n2[0]), color_edge, edge_width)

    scale = 1

    for k, v in node_neighbors.items():
        n1 = k
        cv2.circle(img, (int(n1[1]) * scale, int(n1[0]) * scale), 2, (255, 0, 0), -1)

    cp, _ = locate_stacking_road(node_neighbors)

    for k, v in cp.items():
        e1 = k[0]
        e2 = k[1]

        if draw_intersection == True:
            cv2.line(img, (int(e1[0][1]), int(e1[0][0])), (int(e1[1][1]), int(e1[1][0])), (0, 255, 0), edge_width)
            cv2.line(img, (int(e2[0][1]), int(e2[0][0])), (int(e2[1][1]), int(e2[1][0])), (0, 0, 255), edge_width)

    Image.fromarray(img).save(save_file)


def detect_local_minima(arr, mask, threshold=0.5):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    """
    fit（适合）结构元中所有像素都和图像区域重合
    hit（击中）结构元中存在像素与图像区域有重合
    侵蚀（erosion）	对于任意图像中任一点(x,y),g(x; y) = 1 如果结构元素fit（适合），否则g(x; y) = 0，
    """
    #     fit（适合）结构元中所有像素都和图像区域重合
    #     hit（击中）结构元中存在像素与图像区域有重合
    #     侵蚀（erosion）	对于任意图像中任一点(x,y),g(x; y) = 1 如果结构元素fit（适合），否则g(x; y) = 0，

    # fit: all pixels in the structure element overlap with the image area
    # hit: there are pixels in the structure element that overlap with the image area
    # erosion: for any point (x, y) in any image, g(x; y) = 1 if the structure element fits, otherwise g(x; y) = 0,

    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where((detected_minima & (mask > threshold)))


#return np.where(detected_minima)


def DrawKP(imagegraph, filename, imagesize=256, max_degree=6):
    vertexness = imagegraph[0, :, :].reshape((imagesize, imagesize))

    for i in range(max_degree):
        vertexness = np.maximum(vertexness, imagegraph[2 + 4 * i, :, :].reshape((imagesize, imagesize)))

    kp = np.copy(vertexness)

    smooth_kp = gaussian_filter(np.copy(kp), 1)

    smooth_kp = smooth_kp / max(np.amax(smooth_kp), 0.001)

    Image.fromarray((smooth_kp * 255.0).astype(np.uint8)).save(filename)


# Main function

# 该方法的主要流程是从图像中检测顶点和边，构建图形，并对图形进行优化和可视化
# main process: detect vertices and edges from the image, build the graph, and optimize and visualize the graph
def DecodeAndVis(imagegraph, filename, imagesize=256, max_degree=6, thr=0.5, edge_thr=0.5, snap=False, kp_limit=500,
                 drop=True, use_graph_refine=True, testing=False, spacenet=False, angledistance_weight=100,
                 snap_dist=15):
    # At the very begining of the training, the vertexness output can be very noisy.
    # The decoder algorithm may find too many keypoints (vertices) and run very slowly.
    # To avoid this slowdown, we limit the total number of the keypoints during training.
    # print("decodeandvis, imagegraph shape:",imagegraph.shape) [26, 352, 352]
    kp_limit = 10000000
    if imagesize < 600:
        kp_limit = 500

    if testing:
        kp_limit = 10000000

    # Create numpy arrays for visualization.
    # 顶点捕捉
    if snap:  # full black
        rgb = np.zeros((imagesize * 4, imagesize * 4, 3), dtype=np.uint8)
        rgb2 = np.zeros((imagesize * 4, imagesize * 4, 3), dtype=np.uint8)
    else:  # full white
        rgb = 255 * np.ones((imagesize * 4, imagesize * 4, 3), dtype=np.uint8)
        rgb2 = 255 * np.ones((imagesize * 4, imagesize * 4, 3), dtype=np.uint8)

    # Step-1: Find vertices
    # Step-1 (a): Find vertices through local minima detection.
    """extract feature map"""
    vertexness = imagegraph[:, :, 0].reshape((imagesize, imagesize))

    # Gaussian blur smoothing
    kp = np.copy(vertexness.cpu().numpy())
    smooth_kp = gaussian_filter(np.copy(kp), 1)
    smooth_kp = smooth_kp / max(np.amax(smooth_kp), 0.001)  # normalize

    keypoints = detect_local_minima(-smooth_kp, smooth_kp, thr)

    cc = 0  # counter

    # Step-1 (b): There could be a case where the local minima detection algorithm fails
    # to detect some of the vertices.
    # For example, we have links a<-->b and b<-->c but b is missing.
    # In this case, we use the edges a-->b and b<--c to recover b.
    #
    # To do so, we locate the endpoint of each edge (from the detected vertices so far.),
    # draw all the endpoints on a numpy array (a 2D image), blur it, and use the same minima
    # detection algorithm to find vertices.
    #
    edgeEndpointMap = np.zeros((imagesize, imagesize))

    for i in range(len(keypoints[0])):
        if cc > kp_limit:
            break
        cc += 1

        x, y = keypoints[0][i], keypoints[1][i]

        for j in range(max_degree):

            """
            每个像素点（或顶点）具有多个通道，这些通道的布局如下：
            第0个通道存储顶点特征值（vertexness）。
            从第2个通道开始，每4个通道一组，存储一条边的特征信息：
            第 2 + 4 * j 个通道存储第 j 条边的边特征值（edgeness）。
            第 2 + 4 * j + 1 个通道存储第 j 条边的附加特征（通常是边的权重或相关信息）。
            第 2 + 4 * j + 2 个通道存储第 j 条边的x方向偏移量。
            第 2 + 4 * j + 3 个通道存储第 j 条边的y方向偏移量。
            """
            if imagegraph[x, y, 2 + 4 * j + 2] * imagegraph[x, y, 0] > thr * thr:  # or thr < 0.2:
                """
                边特征值（edgeness）： 这是一个反映边存在可能性的值。值越大，表示边存在的可能性越高。
                顶点特征值（vertexness）：这是一个反映顶点存在可能性的值。值越大，表示顶点存在的可能性越高。
                乘积：通过将边特征值和顶点特征值相乘，可以综合评估该边的可靠性。如果这两个值的乘积大于设定的阈值 thr * thr，则认为这条边有效。
                """
                x1 = int(x + vector_norm * imagegraph[x, y, 2 + 4 * j + 2])
                y1 = int(y + vector_norm * imagegraph[x, y, 2 + 4 * j + 3])

                if 0 <= x1 < imagesize and 0 <= y1 < imagesize:
                    edgeEndpointMap[x1, y1] = imagegraph[x, y, 2 + 4 * j + 2] * imagegraph[x, y, 0]
                    """
                    检查计算得到的端点是否在图像范围内：
                    确保 x1 和 y1 在有效范围 [0, imagesize) 之内。
                    如果端点有效，将其记录到 edgeEndpointMap 中：
                    记录的值是边的特征值（edgeness）和顶点特征值（vertexness）的乘积
                    """

    # gaussian filter
    edgeEndpointMap = gaussian_filter(edgeEndpointMap, 3)
    edgeEndpoints = detect_local_minima(-edgeEndpointMap, edgeEndpointMap, thr * thr * thr)

    # Step-1 (c): Create rtree index to speed up the queries.
    # We need to insert the vertices detected in Step-1(a) and Step-1(b) to the rtree.
    # For the vertices detected in Step-1(b), to avoid duplicated vertices, we only
    # insert them when there are no nearby vertices around them.
    #
    idx = index.Index()

    if snap:
        cc = 0

        # Insert keypoints to the rtree
        for i in range(len(keypoints[0])):
            if cc > kp_limit:
                break

            x, y = keypoints[0][i], keypoints[1][i]

            idx.insert(i, (x - 1, y - 1, x + 1, y + 1))

            cc += 1

        """edge endpoint"""
        # Insert edge endpoints (the other vertex of the edge) to the rtree
        # To avoid duplicated vertices, we only insert the vertex when there is no
        # other vertex nearby.
        for i in range(len(edgeEndpoints[0])):
            if cc > kp_limit * 2:
                break

            x, y = edgeEndpoints[0][i], edgeEndpoints[1][i]

            """
            检查边端点周围是否有其他顶点存在，如果没有则将其插入R树中。
            这样做是为了避免重复顶点的插入。"""
            candidates = list(idx.intersection((x - 5, y - 5, x + 5, y + 5)))

            if len(candidates) == 0:
                idx.insert(i + len(keypoints[0]), (x - 1, y - 1, x + 1, y + 1))  # R树索引
            cc += 1

    # Step-2 Connect the vertices to build a graph.

    # endpoint lookup
    neighbors = {}

    # traversal and connect vertices
    cc = 0
    for i in range(len(keypoints[0])):

        if cc > kp_limit:
            break

        x, y = keypoints[0][i], keypoints[1][i]

        for j in range(max_degree):
            if imagegraph[x, y, 2 + 4 * j + 2] * imagegraph[x, y, 0] > thr * edge_thr and imagegraph[x, y, 2 + 4 * j + 2] > edge_thr:

                # 边的端点坐标
                x1 = int(x + vector_norm * imagegraph[x, y, 2 + 4 * j + 2])
                y1 = int(y + vector_norm * imagegraph[x, y, 2 + 4 * j + 3])

                skip = False

                l = vector_norm * np.sqrt(
                    imagegraph[x, y, 2 + 4 * j + 2].cpu().numpy() * imagegraph[x, y, 2 + 4 * j + 2].cpu().numpy() +
                    imagegraph[x, y, 2 + 4 * j + 3].cpu().numpy() * imagegraph[x, y, 2 + 4 * j + 3].cpu().numpy()
                )

                if snap == True:
                    """
                    顶点捕捉
                    """
                    # We look for a candidate vertex to connect through three passes
                    # Here, we use d(a-->b) to represent the distance metric for edge a-->b .
                    # Pass-1 For a link a<-->b, we connect them only if d(a-->b) + d(a<--b) <= snap_dist.
                    # Pass-2 (relaxed) For a link a<-->b, we connect them only if 2*d(a-->b) <= snap_dist or 2*d(a<--b) <= snap_dist.
                    # Pass-3 (more relaxed) For a link a<-->b, we connect them only if d(a-->b) <= snap_dist or d(a<--b) <= snap_dist.
                    #
                    # In Pass-1 and Pass-2, we only consider the keypoints detected directly by the minima detection algorithm (Step-1(a)).
                    # In Pass-3, we only consider the edge end points detected in Step-1(b)
                    #
                    best_candidate = -1  # init
                    min_distance = snap_dist  #15.0  # 初始化最小距离

                    """
                    Candidate是从R树索引返回的索引值。
                    Candidate < len(keypoints)：如果 candidate 的值小于 keypoints 的长度，表示它是 keypoints 中的索引，即一个关键点。
                                    """
                    candidates = list(idx.intersection((x1 - 20, y1 - 20, x1 + 20, y1 + 20)))

                    # Pass-1 (restrict distance metric)
                    for candidate in candidates:
                        # only snap to keypoints
                        if candidate >= len(keypoints[0]):  # edge points
                            continue

                        # get the coordinate
                        if candidate < len(keypoints[0]):
                            x_c = keypoints[0][candidate]
                            y_c = keypoints[1][candidate]
                        else:
                            x_c = edgeEndpoints[0][candidate - len(keypoints[0])]
                            y_c = edgeEndpoints[1][candidate - len(keypoints[0])]

                        d = distance((x_c, y_c), (x1, y1))

                        if d > l:
                            continue

                        # vector from the edge endpoint (the other side of the edge) to the current vertex.
                        v0 = (x - x_c, y - y_c)

                        min_sd = angledistance_weight

                        for jj in range(max_degree):
                            if imagegraph[x_c, y_c, 2 + 4 * jj] * imagegraph[x_c, y_c, 0] > thr * edge_thr and \
                                    imagegraph[x_c, y_c, 2 + 4 * jj] > edge_thr:
                                vc = (vector_norm * imagegraph[x_c, y_c, 2 + 4 * jj + 2],
                                      vector_norm * imagegraph[x_c, y_c, 2 + 4 * jj + 3])

                                # cosine distance
                                vc = (vc[0].cpu(), vc[1].cpu())
                                vc = (vc[0].numpy(), vc[1].numpy())
                                ad = 1.0 - anglediff(v0, vc)  # line 20
                                ad = ad * angledistance_weight

                                if ad < min_sd:
                                    min_sd = ad

                        d = d + min_sd

                        # cosine distance between the original output edge direction and the edge direction after snapping.
                        v1 = (x_c - x, y_c - y)
                        v2 = (x1 - x, y1 - y)
                        # cosine distance
                        ad = 1.0 - anglediff(v1, v2)  # -1 to 1 # angle difference

                        d = d + ad * angledistance_weight  # 0.15 --> 15 degrees

                        if d < min_distance:
                            min_distance = d
                            best_candidate = candidate

                    # Pass-2 (relax the distance metric)
                    min_distance = snap_dist  #15.0
                    # only need the second pass when there is no good candidate found in the first pass.
                    if best_candidate == -1:
                        for candidate in candidates:
                            # only snap to keypoints
                            if candidate >= len(keypoints[0]):
                                continue

                            if candidate < len(keypoints[0]):
                                x_c = keypoints[0][candidate]
                                y_c = keypoints[1][candidate]
                            else:
                                x_c = edgeEndpoints[0][candidate - len(keypoints[0])]
                                y_c = edgeEndpoints[1][candidate - len(keypoints[0])]

                            d = distance((x_c, y_c), (x1, y1))
                            if d > l * 0.5:  # 放宽距离限制
                                continue

                            # cosine distance between the original output edge direction and the edge direction after snapping.
                            v1 = (x_c - x, y_c - y)
                            v2 = (x1 - x, y1 - y)

                            ad = 1.0 - anglediff(v1, v2)  # -1 to 1
                            d = d + ad * angledistance_weight * 2  # 0.15 --> 30

                            if d < min_distance:
                                min_distance = d
                                best_candidate = candidate

                    # Pass-3 (relax the distance metric even more)
                    if best_candidate == -1:
                        for candidate in candidates:
                            # only snap to edge endpoints
                            if candidate < len(keypoints[0]):
                                continue

                            if candidate < len(keypoints[0]):
                                x_c = keypoints[0][candidate]
                                y_c = keypoints[1][candidate]
                            else:
                                x_c = edgeEndpoints[0][candidate - len(keypoints[0])]
                                y_c = edgeEndpoints[1][candidate - len(keypoints[0])]

                            d = distance((x_c, y_c), (x1, y1))
                            if d > l:
                                continue

                            v1 = (x_c - x, y_c - y)
                            v2 = (x1 - x, y1 - y)

                            ad = 1.0 - anglediff(v1, v2)  # -1 to 1
                            d = d + ad * angledistance_weight  # 0.15 --> 15

                            if d < min_distance:
                                min_distance = d
                                best_candidate = candidate

                    # update best candidate point
                    if best_candidate != -1:
                        if best_candidate < len(keypoints[0]):
                            x1 = keypoints[0][best_candidate]
                            y1 = keypoints[1][best_candidate]
                        else:
                            # edge Endpoint case
                            x1 = edgeEndpoints[0][best_candidate - len(keypoints[0])]
                            y1 = edgeEndpoints[1][best_candidate - len(keypoints[0])]
                    else:
                        skip = True

                # visualization
                c = int(imagegraph[x, y, 2 + 4 * j] * 200.0) + 55  # in [55, 255]
                color = (c, c, c)

                if not snap:
                    color = (255 - c, 255 - c, 255 - c)

                w = 2

                # draw the edges and add them in 'neighbors'
                if skip == False or drop == False:

                    nk1 = (x1, y1)
                    nk2 = (x, y)

                    if nk1 != nk2:
                        if nk1 in neighbors:
                            if nk2 in neighbors[nk1]:
                                pass
                            else:
                                neighbors[nk1].append(nk2)
                        else:
                            neighbors[nk1] = [nk2]

                        if nk2 in neighbors:
                            if nk1 in neighbors[nk2]:
                                pass
                            else:
                                neighbors[nk2].append(nk1)
                        else:
                            neighbors[nk2] = [nk1]

                    cv2.line(rgb, (y1 * 4, x1 * 4), (y * 4, x * 4), color, w)
        cc += 1

    # Step-3 Some graph refinement post-processing passes.
    # We also apply them to other methods in the evaluation.
    #

    spurs_thr = 50
    isolated_thr = 200

    # spacenet's tiles are small
    if spacenet:
        spurs_thr = 25
        isolated_thr = 100

    if imagesize < 400:
        spurs_thr = 25
        isolated_thr = 100

    if use_graph_refine:
        graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)

        _vis(neighbors, filename + "_norefine_bk.png", size=imagesize)

        rc = 100
        while rc > 0:
            if spacenet:
                isolated_thr = 0
                spurs_thr = 0

            if imagesize < 400:
                spurs_thr = 25
                isolated_thr = 100

            graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))

        if spacenet:
            spurs_thr = 25
            isolated_thr = 100

        if imagesize < 400:
            spurs_thr = 25
            isolated_thr = 100

        graph = graph_shave(graph, spurs_thr=spurs_thr)
    else:
        graph = neighbors  # iter until refine finish

    _vis(graph, filename + "_refine_bk.png", size=imagesize, draw_intersection=True)

    cc = 0

    if not snap:

        for i in range(len(keypoints[0])):
            if cc > kp_limit:
                break

            x, y = keypoints[0][i], keypoints[1][i]
            print("keypoints", x, y)

            cv2.circle(rgb, (y * 4, x * 4), 5, (255, 0, 0), -1)
            cc += 1

            d = 0

            for j in range(max_degree):

                #if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr * thr: # or thr < 0.2:
                if imagegraph[x, y, 2 + 4 * j] * imagegraph[x, y, 0] > thr * 0.5:  # or thr < 0.2:
                    d += 1

            color = (255, 0, 0)
            # set color according to the depth
            if d == 2:
                color = (0, 255, 0)
            if d == 3:
                color = (0, 128, 128)

            if d >= 4:
                color = (0, 0, 255)

            cv2.circle(rgb2, (y * 4, x * 4), 8, color, -1)

    else:
        for i in range(len(keypoints[0])):
            if cc > kp_limit:
                break

            x, y = keypoints[0][i], keypoints[1][i]

            cv2.circle(rgb, (y * 4, x * 4), 5, (255, 0, 0), -1)
            cc += 1

            d = 0

            for j in range(max_degree):
                if imagegraph[x, y, 2 + 4 * j] * imagegraph[x, y, 0] > thr * 0.5:  # or thr < 0.2:
                    d += 1

            color = (255, 0, 0)
            if d == 2:
                color = (0, 255, 0)
            if d == 3:
                color = (0, 128, 128)

            if d >= 4:
                color = (0, 0, 255)

            cv2.circle(rgb2, (y * 4, x * 4), 8, color, -1)

        for i in range(len(edgeEndpoints[0])):
            x, y = edgeEndpoints[0][i], edgeEndpoints[1][i]
            cv2.circle(rgb, (y * 4, x * 4), 3, (0, 255, 0), -1)

    # ImageGraphVisIntersection(imagegraph, filename, imagesize=imagesize)
    # print("last step rgb=", rgb.shape)  # (3, 1408, 1408)
    Image.fromarray(rgb).save(filename + "_imagegraph.png")
    Image.fromarray(rgb2).save(filename + "_intersection_node.png")

    with open(f"{filename}_graph.p", "wb") as f:
        pickle.dump(graph, f)

    return graph
