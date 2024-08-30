import cv2
from rtree import index
from common import *
import math
import numpy as np
# import scipy.misc
# from PIL import Image
# import sys
# import pickle


# latitude, longitude
def GPSDistance(p1, p2):
    a = p1[0] - p2[0]
    b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))

    return math.sqrt(a * a + b * b)


def graphInsert(node_neighbor, n1key, n2key):
    if n1key != n2key:  # 非同节点
        if n1key in node_neighbor:  # n1 在邻接表中
            if n2key in node_neighbor[n1key]:
                pass
            else:
                node_neighbor[n1key].append(n2key)
        else:
            node_neighbor[n1key] = [n2key]  # 不在则创建新的邻接表

        if n2key in node_neighbor:
            if n1key in node_neighbor[n2key]:
                pass
            else:
                node_neighbor[n2key].append(n1key)
        else:
            node_neighbor[n2key] = [n1key]

    return node_neighbor


# 通过插值法调整节点的密度
def graphDensify(node_neighbor, density=0.00020):
    visited = []

    new_node_neighbor = {}

    for node, node_nei in node_neighbor.items():
        if len(node_nei) == 1 or len(node_nei) > 2:  # 度数≥1
            if node in visited:
                continue

            # search node_nei

            for next_node in node_nei:  # 邻节点
                if next_node in visited:
                    continue

                node_list = [node, next_node]  # 初始化

                current_node = next_node

                while True:
                    if len(node_neighbor[node_list[-1]]) == 2:  # 当前链的最后一个节点有两个邻居
                        if node_neighbor[node_list[-1]][0] == node_list[-2]:
                            node_list.append(node_neighbor[node_list[-1]][1])
                        else:  # 节点不为2 则结束
                            node_list.append(node_neighbor[node_list[-1]][0])
                    else:
                        break

                for i in range(len(node_list) - 1):
                    if node_list[i] not in visited:
                        visited.append(node_list[i])

                # interpolate
                # partial distance
                pd = [0]

                for i in range(len(node_list) - 1):
                    pd.append(pd[-1] + GPSDistance(node_list[i], node_list[i + 1]))

                interpolate_N = int(pd[-1] / density)  # 需要差值的点的数量

                last_loc = node_list[0]  # 最后的位置为链的起点

                for i in range(interpolate_N):
                    int_d = pd[-1] / (interpolate_N + 1) * (i + 1)
                    for j in range(len(node_list) - 1):
                        if pd[j] <= int_d < pd[j + 1]:
                            a = (int_d - pd[j]) / (pd[j + 1] - pd[j])  # 插值比例

                            loc = ((1 - a) * node_list[j][0] + a * node_list[j + 1][0],
                                   (1 - a) * node_list[j][1] + a * node_list[j + 1][1])

                            new_node_neighbor = graphInsert(new_node_neighbor, last_loc, loc)
                            last_loc = loc

                new_node_neighbor = graphInsert(new_node_neighbor, last_loc, node_list[-1])

    return new_node_neighbor


# GPS坐标 → 像素坐标
# region 是边框的值列表，[min_lat, min_lon, max_lat, max_lon]
def graph2RegionCoordinate(node_neighbor, region):
    new_node_neighbor = {}

    for node, nei in node_neighbor.items():
        loc0 = node  # 当前
        for loc1 in nei:
            # x_0
            x0 = (loc0[1] - region[1]) / (region[3] - region[1]) * 2048
            y0 = (region[2] - loc0[0]) / (region[2] - region[0]) * 2048
            # y_0
            x1 = (loc1[1] - region[1]) / (region[3] - region[1]) * 2048
            y1 = (region[2] - loc1[0]) / (region[2] - region[0]) * 2048

            n1key = (y0, x0)
            n2key = (y1, x1)

            new_node_neighbor = graphInsert(new_node_neighbor, n1key, n2key)

    return new_node_neighbor


# 可视化节点和边
def graphVis2048(node_neighbor, region, filename):
    img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    img = img + 255  # 白色背景

    for node, nei in node_neighbor.items():
        loc0 = node
        for loc1 in nei:
            x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * 2048)
            y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * 2048)
            x1 = int((loc1[1] - region[1]) / (region[3] - region[1]) * 2048)
            y1 = int((region[2] - loc1[0]) / (region[2] - region[0]) * 2048)

            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 2)  # 线段

    for node, nei in node_neighbor.items():
        loc0 = node
        x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * 2048)
        y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * 2048)

        cv2.circle(img, (x0, y0), 3, (0, 0, 255), -1)  # 点

    cv2.imwrite(filename, img)


# 只含道路的分割图像
def graphVis2048Segmentation(node_neighbor, region, filename, size=2048):
    img = np.zeros((size, size), dtype=np.uint8)

    for node, nei in node_neighbor.items():
        loc0 = node
        for loc1 in nei:
            x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)
            y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)
            x1 = int((loc1[1] - region[1]) / (region[3] - region[1]) * size)
            y1 = int((region[2] - loc1[0]) / (region[2] - region[0]) * size)

            cv2.line(img, (x0, y0), (x1, y1), (255, 255, 255), 2)

    cv2.imwrite(filename, img)


# 重叠道路 stacking road 的交点
def graphVisStackingRoad(node_neighbor, region, filename, size=2048):
    img = np.zeros((size, size), dtype=np.uint8)

    crossing_point, adjustment = locate_stacking_road(node_neighbor)

    for ip in crossing_point.values():
        loc0 = ip
        x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)
        y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)

        cv2.circle(img, (x0, y0), 5, (255, 255, 255), -1)

    cv2.imwrite(filename, img)


# 交叉点 十字路口等
def graphVisIntersection(node_neighbor, region, filename, size=2048):
    img = np.zeros((size, size), dtype=np.uint8)

    for node, nei in node_neighbor.iitems():
        loc0 = node

        if len(nei) != 2:
            x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)
            y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)

            cv2.circle(img, (x0, y0), 5, (255, 255, 255), -1)

    cv2.imwrite(filename, img)


# 定位并计算
def locate_stacking_road(graph):
    idx = index.Index()  # R tree

    edges = []

    for n1, v in graph.items():
        for n2 in v:
            if (n1, n2) in edges or (n2, n1) in edges:
                continue

            x1 = min(n1[0], n2[0])
            x2 = max(n1[0], n2[0])

            y1 = min(n1[1], n2[1])
            y2 = max(n1[1], n2[1])

            idx.insert(len(edges), (x1, y1, x2, y2))

            edges.append((n1, n2))

    adjustment = {}

    crossing_point = {}

    for edge in edges:
        n1 = edge[0]  # 边的起点
        n2 = edge[1]  # 边的终点

        x1 = min(n1[0], n2[0])
        x2 = max(n1[0], n2[0])

        y1 = min(n1[1], n2[1])
        y2 = max(n1[1], n2[1])

        candidates = list(idx.intersection((x1, y1, x2, y2)))  # 相交候选边
        # candidates 是与当前边可能相交的候选边的索引列表。
        # _candidate 是当前遍历的候选边的索引。
        for _candidate in candidates:
            # todo mark the overlap point
            candidate = edges[_candidate]

            # 存在公共节点
            if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
                continue

            if intersect(n1, n2, candidate[0], candidate[1]):

                ip = intersectPoint(n1, n2, candidate[0], candidate[1])  # 交点

                if (candidate, edge) not in crossing_point:
                    crossing_point[(edge, candidate)] = ip

                # release points
                d = distance(ip, n1)
                thr = 9.5  # was 5.0
                if d < thr:
                    vec = neighbors_norm(graph, n1, n2)
                    weight = (thr - d) / thr
                    vec = (vec[0] * weight, vec[1] * weight)  # 调整向量

                    if n1 not in adjustment:
                        adjustment[n1] = [vec]
                    else:
                        adjustment[n1].append(vec)

                d = distance(ip, n2)
                if d < thr:
                    vec = neighbors_norm(graph, n2, n1)
                    weight = (thr - d) / thr
                    vec = (vec[0] * weight, vec[1] * weight)

                    if n2 not in adjustment:
                        adjustment[n2] = [vec]
                    else:
                        adjustment[n2].append(vec)

                c1 = candidate[0]
                c2 = candidate[1]

                d = distance(ip, c1)
                if d < thr:
                    vec = neighbors_norm(graph, c1, c2)
                    weight = (thr - d) / thr
                    vec = (vec[0] * weight, vec[1] * weight)

                    if c1 not in adjustment:
                        adjustment[c1] = [vec]
                    else:
                        adjustment[c1].append(vec)

                d = distance(ip, c2)
                if d < thr:
                    vec = neighbors_norm(graph, c2, c1)
                    weight = (thr - d) / thr
                    vec = (vec[0] * weight, vec[1] * weight)

                    if c2 not in adjustment:
                        adjustment[c2] = [vec]
                    else:
                        adjustment[c2].append(vec)

    # apply adjustment
    # move 2 pixels each time

    return crossing_point, adjustment


def locate_parallel_road(graph):
    idx = index.Index()

    edges = []

    for n1, v in graph.items():
        for n2 in v:
            if (n1, n2) in edges or (n2, n1) in edges:  # 边已存在
                continue

            x1 = min(n1[0], n2[0])
            x2 = max(n1[0], n2[0])

            y1 = min(n1[1], n2[1])
            y2 = max(n1[1], n2[1])

            idx.insert(len(edges), (x1, y1, x2, y2))

            edges.append((n1, n2))

    adjustment = {}

    crossing_point = {}

    parallel_road = []

    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]

        if distance(n1, n2) < 10:  # 边的长度小于10时不检测，避免噪声
            continue

        x1 = min(n1[0], n2[0]) - 20
        x2 = max(n1[0], n2[0]) + 20

        y1 = min(n1[1], n2[1]) - 20
        y2 = max(n1[1], n2[1]) + 20

        candidates = list(idx.intersection((x1, y1, x2, y2)))  # 查找候选边

        for _candidate in candidates:
            # todo mark the overlap point
            candidate = edges[_candidate]

            # 有交点则不可能平行
            if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
                continue

            # 是否找到公共的邻居
            flag = False

            # 候选边的起点和终点都在邻居中
            for nei in graph[n1]:
                if candidate[0] in graph[nei]:
                    flag = True
                    continue

                if candidate[1] in graph[nei]:
                    flag = True
                    continue

            for nei in graph[n2]:
                if candidate[0] in graph[nei]:
                    flag = True
                    continue

                if candidate[1] in graph[nei]:
                    flag = True
                    continue

            if flag:
                continue

            # 余弦相似度
            p = abs(neighbors_cos(graph, (0, 0), (n2[0] - n1[0], n2[1] - n1[1]),
                                  (candidate[1][0] - candidate[0][0], candidate[1][1] - candidate[0][1])))

            if p > 0.985:  # 判定平行
                if n1 not in parallel_road:
                    parallel_road.append(n1)

    return parallel_road


def apply_adjustment(graph, adjustment):
    current_graph = graph
    counter = 0

    for k, v in adjustment.items():
        # print(k)
        new_graph = {}

        vec = [0, 0]  # init

        for vv in v:
            vec[0] += vv[0]  # x 分量
            vec[1] += vv[1]

        vl = vec[0] * vec[0] + vec[1] * vec[1]

        vl = np.sqrt(vl)  # length

        if vl == 0:
            continue

        if vl > 1.0:
            vec[0] /= vl  # normalize
            vec[1] /= vl

        for l in [1.5, 1.0]:  # 遍历缩放因子

            # new_k = (k[0] + int(vec[0]*l), k[1] + int(vec[1]*l))
            new_k = (k[0] + (vec[0] * l), k[1] + (vec[1] * l))

            if new_k == k:  # 新旧位置相同
                continue

            if new_k not in current_graph:  # 新位置不在当前图中，在第一轮循环中删除当前，添加新节点

                neighbors = list(current_graph[k])

                del current_graph[k]

                current_graph[new_k] = neighbors

                for nei in neighbors:
                    new_nei = []

                    for n in current_graph[nei]:  # 第二轮循环， 处理邻居的邻居
                        if n == k:
                            new_nei.append(new_k)
                        else:
                            new_nei.append(n)

                    current_graph[nei] = new_nei

                # print(k, "-->", new_k)

                counter += 1

                break
            else:
                continue

    print("adjusted ", counter, " nodes")

    return current_graph, counter

    pass


def graph_move_node(graph, old_n, new_n):
    nei = list(graph[old_n])
    del graph[old_n]

    graph[new_n] = nei

    for nn in nei:
        for i in range(len(graph[nn])):
            if graph[nn][i] == old_n:
                graph[nn][i] = new_n

    return graph


def apply_adjustment_delete_closeby_nodes(graph, adjustment):
    # delete the node and push its two neighbors closer ...

    # thr = 9.5
    thr = 9.5
    for k, v in adjustment.items():
        if len(v) >= 4:  # duplicated ...调整向量数量>4
            ds = []
            for vv in v:
                ds.append((1.0 - distance(vv, (0, 0))) * thr)  # 计算调整向量的距离并排序
            sorted(ds)
            gap = sum(ds[0:4]) / 2.0  # 前四个距离向量的平均值

            # delete the node and push its two neighbors closer ...
            if gap < 12 and len(graph[k]) == 2:
                nei1 = graph[k][0]
                nei2 = graph[k][1]

                del graph[k]
                print("delete a node", k)  # 删除过于密集的点

                for i in range(len(graph[nei1])):
                    if graph[nei1][i] == k:
                        graph[nei1][i] = nei2

                for i in range(len(graph[nei2])):
                    if graph[nei2][i] == k:
                        graph[nei2][i] = nei1

                # move nei1/nei2 closer?
                if nei1 not in adjustment:
                    vec = neighbors_norm(graph, k, nei1)
                    new_nei1 = (nei1[0] + vec[0] * 5.0, nei1[1] + vec[1] * 5.0)
                    graph = graph_move_node(graph, nei1, new_nei1)

                if nei2 not in adjustment:
                    vec = neighbors_norm(graph, k, nei2)
                    new_nei2 = (nei2[0] + vec[0] * 5.0, nei2[1] + vec[1] * 5.0)
                    graph = graph_move_node(graph, nei2, new_nei2)

    return graph


def graphGroundTruthPreProcess(graph):
    cp = {}
    for it in range(40):  # was 8
        cp, adj = locate_stacking_road(graph)
        if it % 5 == 0 and it != 0:
            # 5次迭代删除一次过近的节点
            graph = apply_adjustment_delete_closeby_nodes(graph, adj)
        else:
            graph, c = apply_adjustment(graph, adj)
            if c == 0:  # 直到没有节点被调整
                break

    sample_points = {'parallel_road': locate_parallel_road(graph), 'complicated_intersections': []}

    for k, v in graph.items():
        degree = len(v)
        if degree > 4:
            sample_points['complicated_intersections'].append(k)

    sample_points['overpass'] = []

    for k, v in cp.items():
        sample_points['overpass'].append((int(v[0]), int(v[1])))

    # 返回decoder 中的 字典
    return graph, sample_points
#
    x


""" 
遍历缩放因子？
adjust 和move 调用的区别？


"""
