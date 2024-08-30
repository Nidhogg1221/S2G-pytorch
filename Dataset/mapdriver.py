import math
import os.path
import xml.etree.ElementTree
import requests
import shutil
from time import sleep
"""
from scipy.ndimage import rotate
from skimage.transform import resize
import imageio.v2 as imageio
import scipy.misc
import pickle
import socket
from PIL import Image
import cv2
import sys
import numpy as np
from subprocess import Popen
import time
"""

ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0  # 经纬度 → 米

img_cache = {}


# 经纬度 → 米
# 墨卡托投影
def lonLatToMeters(lon, lat):
    mx = lon * ORIGIN_SHIFT / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * ORIGIN_SHIFT / 180.0
    return mx, my


# 米 → 经纬度
def metersToLonLat(mx, my):
    lon = (mx / ORIGIN_SHIFT) * 180.0
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lon, lat

# load and parse OSM data
class OSMLoader:
    def __init__(self, region, noUnderground=False, osmfile=None, includeServiceRoad=False):

        # sub_range = str(region[1]) + "," + str(region[0]) + "," + str(region[3]) + "," + str(region[2])
        sub_range = str(region[0]) + "," + str(region[1]) + "," + str(region[2]) + "," + str(region[3])

        config_dir = r"C:\Users\USER\PycharmProjects\pythonProject\Sat2Graph-master\kr_prepare_dataset\config"
        osm_filename = "uam_data.osm"
        osm_filepath = os.path.join(config_dir, osm_filename)

        if not os.path.exists(osm_filepath):
            while not os.path.exists(osm_filepath):
                response = requests.get(f"http://overpass-api.de/api/map?bbox={sub_range}")
                with open(osm_filename, 'wb') as f:
                    f.write(response.content)
                shutil.move(osm_filename, osm_filepath)
                if not os.path.exists(osm_filepath):
                    print("Error. Wait for one minute")
                    sleep(60)
            filename = osm_filepath
            print("OSM file downloaded successfully")
            print("range:", sub_range)
        else:
            filename = osm_filepath

        # # 过滤复杂道路
        roadForMotorDict = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', "highway", "bridge", "unclassified, "}
        roadForMotorBlackList = {'None', 'pedestrian', 'footway', 'bridleway', 'steps', 'path', 'sidewalk', 'cycleway',
                                 'proposed', 'construction', 'bus_stop', 'crossing', 'elevator',
                                 'emergency_access_point', 'escape', 'give_way'}

        # roadForMotorDict = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'pedestrian', 'footway', 'bridleway', 'steps', 'path', 'sidewalk', 'cycleway',
        #                          'proposed', 'construction', 'bus_stop', 'crossing', 'elevator',
        #                          'emergency_access_point', 'escape', 'give_way'}
        # roadForMotorBlackList = {'None'}

        # 读取osm文件
        # 第一行 xml 文件头
        mapxml = xml.etree.ElementTree.parse(filename).getroot()

        # 找到所有的node, way, relation
        nodes = mapxml.findall('node')
        ways = mapxml.findall('way')
        relations = mapxml.findall('relation')

        self.nodedict = {}
        self.waydict = {}
        self.roadlist = []
        self.roaddict = {}

        """ 
        这是一个字典，用于将边（edge）的节点对（tuple of node IDs）映射到边的唯一ID。
        目的是为了快速查找每个边对应的唯一ID，以便在图中进行边的管理。"""
        self.edge2edgeid = {}

        """
        这是一个字典，用于将唯一的边ID映射回边（edge）的节点对（tuple of node IDs）。
        目的是为了根据边ID快速查找具体的边，以便在需要时访问边的详细信息。"""
        self.edgeid2edge = {}

        """
        这是一个字典，用于存储每个边的属性信息。
        键是边的唯一ID，值是一个字典，包含该边的详细属性（例如宽度、车道数、道路类型等）。"""
        self.edgeProperty = {}
        self.edgeId = 0
        way_c = 0

        # 边界
        self.minlat = float(mapxml.find('bounds').get('minlat'))
        self.maxlat = float(mapxml.find('bounds').get('maxlat'))
        self.minlon = float(mapxml.find('bounds').get('minlon'))
        self.maxlon = float(mapxml.find('bounds').get('maxlon'))

        # traversal all nodes and save them in a dictionary
        for anode in nodes:
            tmp = {}
            tmp['node'] = anode
            tmp['lat'] = float(anode.get('lat'))
            tmp['lon'] = float(anode.get('lon'))
            tmp['to'] = {}
            tmp['from'] = {}

            self.nodedict.update({anode.get('id'): tmp})

        self.buildings = []

        # initialize the way network, remove some useless ways
        for away in ways:
            nds = away.findall('nd')
            highway = 'None'
            lanes = -1
            width = -1
            layer = 0

            hasLane = False
            hasWidth = False
            fromMassGIS = False

            parking = False

            oneway = 0

            isBuilding = False

            building_height = 6

            cycleway = "none"

            info_dict = {}

            # deal with tags
            for atag in away.findall('tag'):
                info_dict[atag.get('k')] = atag.get('v')

                if atag.get('k').startswith("cycleway"):
                    cycleway = atag.get('v')

                if atag.get('k') == 'building':
                    isBuilding = True

                if atag.get('k') == 'highway':
                    highway = atag.get('v')
                if atag.get('k') == 'lanes':
                    try:
                        lanes = float(atag.get('v').split(';')[0])
                    except ValueError:
                        lanes = -1

                    hasLane = True
                if atag.get('k') == 'width':
                    #print(atag.get('v'))
                    try:
                        width = float(atag.get('v').split(';')[0].split()[0])
                    except ValueError:
                        width = -1

                    hasWidth = True
                if atag.get('k') == 'layer':
                    try:
                        layer = int(atag.get('v'))
                    except ValueError:
                        print("ValueError for layer", atag.get('v'))
                        layer = -1

                if atag.get('k') == 'source':
                    if 'massgis' in atag.get('v'):
                        fromMassGIS = True

                if atag.get('k') == 'amenity':
                    if atag.get('v') == 'parking':
                        parking = True

                if atag.get('k') == 'service':
                    if atag.get('v') == 'parking_aisle':
                        parking = True

                if atag.get('k') == 'service':
                    if atag.get('v') == 'driveway':
                        parking = True

                if atag.get('k') == 'oneway':
                    if atag.get('v') == 'yes':
                        oneway = 1
                    if atag.get('v') == '1':
                        oneway = 1
                    if atag.get('v') == '-1':
                        oneway = -1

                if atag.get('k') == 'height':
                    try:
                        building_height = float(atag.get('v').split(' ')[0])
                    except ValueError:
                        print(atag.get('v'))

                if atag.get('k') == 'ele':
                    try:
                        building_height = float(atag.get('v').split(' ')[0]) * 3
                    except ValueError:
                        print(atag.get('v'))

            # 调整路径宽度
            if width == -1:
                if lanes == -1:
                    width = 6.6
                else:
                    if lanes == 1:
                        width = 6.6
                    else:
                        width = 3.7 * lanes

            if lanes != -1:
                if width > lanes * 3.7 * 2:
                    width = width / 2
                if lanes == 1:
                    width = 6.6
                else:
                    width = lanes * 3.7

            if noUnderground:
                if layer < 0:
                    continue

            if isBuilding:
                idlink = []
                for anode in away.findall('nd'):
                    refid = anode.get('ref')
                    idlink.append(refid)

                    self.buildings.append(
                        [[(self.nodedict[x]['lat'], self.nodedict[x]['lon']) for x in idlink], building_height])

            #if highway in roadForMotorDict: #and hasLane and hasWidth and fromMassGIS:
            #if highway not in roadForMotorBlackList:
            #if highway in roadForMotorDict:

            #if highway not in roadForMotorBlackList and parking == False:
            # 过滤非机动车道，停车道，服务道路等
            if highway not in roadForMotorBlackList and (
                    includeServiceRoad == True or parking == False):  # include parking roads!

                idlink = []
                for anode in away.findall('nd'):
                    refid = anode.get('ref')
                    idlink.append(refid)

                for i in range(len(idlink) - 1):
                    link1 = (idlink[i], idlink[i + 1])
                    link2 = (idlink[i + 1], idlink[i])

                    if link1 not in self.edge2edgeid.keys():
                        self.edge2edgeid[link1] = self.edgeId
                        self.edgeid2edge[self.edgeId] = link1
                        self.edgeProperty[self.edgeId] = {"width": width, "lane": lanes, "layer": layer,
                                                          "roadtype": highway, "cycleway": cycleway,
                                                          "info": dict(info_dict)}
                        self.edgeId += 1

                    if link2 not in self.edge2edgeid.keys():
                        self.edge2edgeid[link2] = self.edgeId
                        self.edgeid2edge[self.edgeId] = link2
                        self.edgeProperty[self.edgeId] = {"width": width, "lane": lanes, "layer": layer,
                                                          "roadtype": highway, "cycleway": cycleway,
                                                          "info": dict(info_dict)}
                        self.edgeId += 1

                if oneway >= 0:
                    for i in range(len(idlink) - 1):
                        self.nodedict[idlink[i]]['to'][idlink[i + 1]] = 1
                        self.nodedict[idlink[i + 1]]['from'][idlink[i]] = 1

                    self.waydict[way_c] = idlink
                    way_c += 1

                idlink.reverse()

                if oneway == -1:
                    for i in range(len(idlink) - 1):
                        self.nodedict[idlink[i]]['to'][idlink[i + 1]] = 1
                        self.nodedict[idlink[i + 1]]['from'][idlink[i]] = 1

                    self.waydict[way_c] = idlink
                    way_c += 1

                if oneway == 0:
                    for i in range(len(idlink) - 1):
                        self.nodedict[idlink[i]]['to'][idlink[i + 1]] = 1
                        self.nodedict[idlink[i + 1]]['from'][idlink[i]] = 1
