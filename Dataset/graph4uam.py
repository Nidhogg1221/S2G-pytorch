import xml.etree.ElementTree as ET
import numpy as np
import cv2
import math
import os
import json

"""
Processing osm files with UAV paths, which contain the lat and lon of interpolated nodes
Output a JSON file containing the boundary box of the interpolated nodes,
 the lat and lon of interpolated nodes and an image of the graph    
"""


def GPSDistance(p1, p2):
    a = p1[0] - p2[0]
    b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))

    return math.sqrt(a * a + b * b)


def graphInsert(node_neighbor, n1key, n2key):
    if n1key != n2key:  # not the same node
        if n1key in node_neighbor:  # n1 in the neighbor list
            if n2key in node_neighbor[n1key]:
                pass
            else:
                node_neighbor[n1key].append(n2key)
        else:
            node_neighbor[n1key] = [n2key]  # create a new list when n1 is not in the neighbor list

        if n2key in node_neighbor:
            if n1key in node_neighbor[n2key]:
                pass
            else:
                node_neighbor[n2key].append(n1key)
        else:
            node_neighbor[n2key] = [n1key]

    return node_neighbor


def graphDensify(node_neighbor, density=0.0018018):
    """
    1 degree latitude is approximately 111000 meters
    density = n / 111000
    interpolate per 50m = 50 / 111000 =  0.00045045
    interpolate per 100m = 100 / 111000 = 0.0009009
    interpolate per 200m = 200 / 111000 = 0.0018018
    interpolate per 500m = 500 / 111000 = 0.0045045
    interpolate per 1000m = 1000 / 111000 = 0.0090090
    """
    visited = []
    new_node_neighbor = {}
    interpolate_Nodes = []  # save the interpolated nodes

    for node, node_nei in node_neighbor.items():
        # pick the node with only one or more than two neighbors
        # because the node with two neighbors is nontrivial
        if len(node_nei) == 1 or len(node_nei) > 2:
            if node in visited:
                continue

            # search node_nei

            for next_node in node_nei:  # 邻节点
                if next_node in visited:
                    continue

                node_list = [node, next_node]  # 初始化链
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

                interpolate_N = int(pd[-1] / density)  # the number of interpolated nodes needed

                last_loc = node_list[0]  # last location is the first vertex of the chain

                for i in range(interpolate_N):
                    int_d = pd[-1] / (interpolate_N + 1) * (i + 1)
                    for j in range(len(node_list) - 1):
                        if pd[j] <= int_d < pd[j + 1]:
                            a = (int_d - pd[j]) / (pd[j + 1] - pd[j])  # 插值比例

                            loc = ((1 - a) * node_list[j][0] + a * node_list[j + 1][0],
                                   (1 - a) * node_list[j][1] + a * node_list[j + 1][1])

                            new_node_neighbor = graphInsert(new_node_neighbor, last_loc, loc)
                            interpolate_Nodes.append(loc)
                            last_loc = loc

                new_node_neighbor = graphInsert(new_node_neighbor, last_loc, node_list[-1])
    print(interpolate_Nodes)  # print the interpolated nodes

    interpolate_node_output_directory = r'C:\Users\USER\PycharmProjects\pythonProject\Sat2Graph-master\kr_prepare_dataset\uam_osm'
    if not os.path.exists(interpolate_node_output_directory):
        os.makedirs(interpolate_node_output_directory)

    filename = f"{interpolate_node_output_directory}/interpolated_Nodes.json"

    # interpolate_nodes_json = [{"lat": loc[0], "lon": loc[1], "lat_n": 1, "lon_n": 1} for loc in interpolate_Nodes]
    interpolate_nodes_json = [{"lat": loc[0], "lon": loc[1]} for loc in interpolate_Nodes]

    with open(filename, 'w') as f:
        json.dump(interpolate_nodes_json, f, ensure_ascii=False, indent=4)

    return new_node_neighbor, interpolate_Nodes


def graphVis2048(node_neighbor, region, filename):
    img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    img = img + 255  # white background

    for node, nei in node_neighbor.items():
        loc0 = node
        for loc1 in nei:
            x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * 2048)
            y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * 2048)
            x1 = int((loc1[1] - region[1]) / (region[3] - region[1]) * 2048)
            y1 = int((region[2] - loc1[0]) / (region[2] - region[0]) * 2048)

            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 2)  # edge

    for node, nei in node_neighbor.items():
        loc0 = node
        x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * 2048)
        y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * 2048)

        cv2.circle(img, (x0, y0), 3, (0, 0, 255), -1)  # vertex

    cv2.imwrite(filename, img)


def Graph4uam(osm_file_path, output_image_path, output_json_path):
    # 解析 OSM 文件
    tree = ET.parse(osm_file_path)
    root = tree.getroot()

    nodes = {}
    ways = []

    # extract the lat and lon of nodes
    for node in root.findall('node'):
        node_id = int(node.get('id'))
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        nodes[node_id] = (lat, lon)

    # extract the nodes of ways
    for way in root.findall('way'):
        nds = []
        for nd in way.findall('nd'):
            ref = int(nd.get('ref'))
            if ref in nodes:
                nds.append(nodes[ref])
        ways.append(nds)

    # 构建邻接表
    node_neighbor = {}
    for way in ways:
        for i in range(len(way) - 1):
            node_neighbor = graphInsert(node_neighbor, way[i], way[i + 1])

    #
    refined_graph, interpolated_nodes = graphDensify(node_neighbor)

    # get the boundary box of the interpolated nodes
    lats = [coord[0] for coord in nodes.values()]
    lons = [coord[1] for coord in nodes.values()]
    minlat, minlon, maxlat, maxlon = min(lats), min(lons), max(lats), max(lons)
    region = [minlat, minlon, maxlat, maxlon]

    # generate the image of the graph
    graphVis2048(refined_graph, region, output_image_path)

    interpolate_node_output_directory = os.path.dirname(output_json_path)
    if not os.path.exists(interpolate_node_output_directory):
        os.makedirs(interpolate_node_output_directory)

    # Create a JSON file containing the boundary box of the interpolated nodes and the lat and lon of interpolated nodes
    interpolate_nodes_json = [{"minlat": minlat, "minlon": minlon}, {"maxlat": maxlat, "maxlon": maxlon}]
    interpolate_nodes_json += [{"lat": loc[0], "lon": loc[1]} for loc in interpolated_nodes]

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(interpolate_nodes_json, f, ensure_ascii=False, indent=4)


# function call
osm_file_path = r'C:\Users\USER\PycharmProjects\pythonProject\Sat2Graph-master\kr_prepare_dataset\uam_osm\uam_sample.osm'  # swicth to the actual path
output_image_path = r'C:\Users\USER\PycharmProjects\pythonProject\Sat2Graph-master\kr_prepare_dataset\uam_osm/output_image.png'  # switch to the actual output img path
output_json_path = r'C:\Users\USER\PycharmProjects\pythonProject\Sat2Graph-master\kr_prepare_dataset\uam_osm\interpolated_Nodes.json'  # switch to the actual output json path

Graph4uam(osm_file_path, output_image_path, output_json_path)
