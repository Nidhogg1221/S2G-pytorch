import sys
import json
from subprocess import Popen
import mapdriver as md
import mapbox as mb
import graph_ops as graphlib
import math
import numpy as np
from PIL import Image
import pickle
import os
import requests
import shutil
from time import sleep

# main
"""
work with json file output from graph4uam.py
which contains the lat and lon of interpolated nodes
"""

dataset_cfg = []
boundary_cfg = []  # read boundary box data from json file
node_cfg = []  # read node data from json file

total_regions = 0

expanded_range = 1024  # meter, for extend the boundary box

tid = 0  # int(sys.argv[1])
tn = 1  # int(sys.argv[2])

# read json file
for name_cfg in sys.argv[1:]:
    with open(name_cfg, "r") as f:
        dataset_cfg_ = json.load(f)
        boundary_cfg.extend(dataset_cfg_[:2])  # read first 2 item
        node_cfg.extend(dataset_cfg_[2:])  # read the rest

first_item = boundary_cfg[0]
second_item = boundary_cfg[1]

# get initial boundary box
min_lat = first_item['minlat']
min_lon = first_item['minlon']
max_lat = second_item['maxlat']
max_lon = second_item['maxlon']
print(f"min_lat: {min_lat}, min_lon: {min_lon}, max_lat: {max_lat}, max_lon: {max_lon}")

# calculate the center of the boundary box
lat_center = (min_lat + max_lat) / 2.0

# calculate the expanded boundary box
expanded_range_lat = expanded_range / (2 * 111111.0)
expanded_range_lon = expanded_range / (2 * 111111.0 * math.cos(math.radians(lat_center)))

# enlarge the boundary box
min_lat -= expanded_range_lat
max_lat += expanded_range_lat
min_lon -= expanded_range_lon
max_lon += expanded_range_lon

print(f"Expanded min_lat: {min_lat}, min_lon: {min_lon}, max_lat: {max_lat}, max_lon: {max_lon}")

sub_range = f"{min_lon},{min_lat},{max_lon},{max_lat}"
"""
when api try to download xml from OSM, the order of param is 
lon, lat, lon, lat.
"""

# download large scale osm
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
    print("osm file downed successfully.")
else:
    filename = osm_filepath

# zoom = 19, area = 1024
# dataset_folder = "0807_0"
# folder_cache = "cache_07_0/"
dataset_folder = "0830"
folder_cache = "cache_0830_0/"

Popen("mkdir %s" % dataset_folder, shell=True).wait()
Popen("mkdir %s" % folder_cache, shell=True).wait()

c = 0
# tiles_needed = 0
total_images = len(node_cfg)
current_image = 0

for item in node_cfg:
    lat = item["lat"]
    lon = item["lon"]
    print(f"node{c} start processing")

    """ calculate the bounding box
    111111.0 is the length of 1 degree in meters
    original size: 2048

    128m: 1067*1067
    256m: 2134*2134
    """
    expanded_range_lat = expanded_range / (2 * 111111.0)
    expanded_range_lon = expanded_range / (2 * 111111.0 * math.cos(math.radians(lat)))

    # 计算边界框的起始和结束的经纬度
    lat_st = lat - expanded_range_lat
    lat_ed = lat + expanded_range_lat
    lon_st = lon - expanded_range_lon
    lon_ed = lon + expanded_range_lon

    # download satellite imagery from vworld
    # set the zoom level according to the latitude
    # max zoom_lever = 19
    zoom_level = 19
    if abs(lat_st) < 33:
        zoom = zoom_level + 1
    else:
        zoom = zoom_level

    """ 
    Just notice we download several tile and merge them together to get the final image.
    So the height wont be limited by the zoom level.
    """

    print(f"node {c}'s coordinate: {lat_st}, {lon_st}, {lat_ed}, {lon_ed}")

    # comment out the image downloading part
    # call vworld API to download satellite imagery
    img, _ = mb.GetMapInRect(lat_st, lon_st, lat_ed, lon_ed, start_lat=lat_st, start_lon=lon_st, zoom=zoom,
                             folder=folder_cache)
    # original imagesize: 8535 * 8535 * 3
    # print(np.shape(img))

    """size of the image"""
    img_size = 2048

    img_pil = Image.fromarray(img.astype(np.uint8))
    # print(f"Image size before resizing: {img_pil.size}") 8535*8535

    img_resized = img_pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
    # print(f"Image size after resizing: {img_resized.size}") 2048*2048

    img = np.array(img_resized)
    Image.fromarray(img).save(f"{dataset_folder}/region_{c}_{lat}_{lon}_sat.png")

    current_image += 1
    print(f"Generated image {current_image}/{total_images}")

    """
    data processing part
    """

    # download openstreetmap
    # read the osm map
    # call mapdriver to download the osm map
    OSMMap = md.OSMLoader([lon_st, lat_st, lon_ed, lat_ed], False, includeServiceRoad=True)

    # neighbor nodes
    node_neighbor = {}  # continuous

    for node_id, node_info in OSMMap.nodedict.items():
        lat = node_info["lat"]
        lon = node_info["lon"]

        n1key = (lat, lon)

        neighbors = []
        for nid in list(node_info["to"].keys()) + list(node_info["from"].keys()):
            if nid not in neighbors:
                neighbors.append(nid)

        for nid in neighbors:
            n2key = (OSMMap.nodedict[nid]["lat"], OSMMap.nodedict[nid]["lon"])

            node_neighbor = graphlib.graphInsert(node_neighbor, n1key, n2key)

    # interpolate the graph
    node_neighbor = graphlib.graphDensify(node_neighbor)

    # gt file
    graphlib.graphVis2048Segmentation(node_neighbor, [lat_st, lon_st, lat_ed, lon_ed],
                                      dataset_folder + "/region_%d_" % c + "gt.png")

    node_neighbor_region = graphlib.graph2RegionCoordinate(node_neighbor, [lat_st, lon_st, lat_ed, lon_ed])
    # pickle file
    prop_graph = dataset_folder + "/region_%d_graph_gt.pickle" % c
    pickle.dump(node_neighbor_region, open(prop_graph, "wb"))

    node_neighbor_refine, sample_points = graphlib.graphGroundTruthPreProcess(node_neighbor_region)

    #.p file
    refine_graph = dataset_folder + "/region_%d_" % c + "refine_gt_graph.p"
    pickle.dump(node_neighbor_refine, open(refine_graph, "wb"))

    # json file
    json.dump(sample_points,
              open(dataset_folder + "/region_%d_" % c + "refine_gt_graph_samplepoints.json", "w"), indent=2)

    c += 1

print("All downloads and processing completed successfully.")

# lat_st, lon_st = 36.386300, 127.3473000
# lat_ed, lon_ed = 36.392400, -127.3532000
