import math
import numpy as np
import os
import scipy.ndimage
from PIL import Image
from subprocess import Popen
from time import time, sleep
import imageio.v2 as imageio
import requests

"""
tile size: 256 * 256 or 512 * 512

"""


# longitude, latitude → tile(x, y)
def lonlat2mapboxTile(lonlat, zoom):
    n = 2 ** zoom

    x = int((lonlat[0] + 180) / 360 * n)
    y = int((1 - math.log(
        math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

    return [x, y]


# longitude, latitude → pixel(x, y)
def lonlat2TilePos(lonlat, zoom):
    n = 2 ** zoom  # tile 的数量

    # 整数坐标
    ix = int((lonlat[0] + 180) / 360 * n)
    iy = int((1 - math.log(
        math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

    x = ((lonlat[0] + 180) / 360 * n)
    y = ((1 - math.log(
        math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

    # 瓦片内的像素位置
    x = int((x - ix) * 512)
    y = int((y - iy) * 512)

    return x, y


def downloadMapBox(zoom, p, outputname):
    url = "https://xdworld.vworld.kr/2d/Satellite/service/{z}/{x}/{y}.jpeg".format(z=zoom, x=p[0], y=p[1])
    filename = "{z}_{x}_{y}.jpeg".format(z=zoom, x=p[0], y=p[1])

    # 确认输出目录存在
    output_dir = os.path.dirname(outputname)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Succ = False

    # print("Attempting to download:", url) 下载每一个小图块都会显示
    retry_timeout = 10

    while not Succ:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filename, 'wb') as file:
                    file.write(response.content)
                os.rename(filename, outputname)
                Succ = True
            else:
                raise Exception("Failed to download, status code:", response.status_code)
        except Exception as e:
            print(e)
            sleep(retry_timeout)
            retry_timeout += 10
            if retry_timeout > 60:
                retry_timeout = 60
            print("Retrying, timeout is", retry_timeout, "seconds")

    return Succ


def downloadMapTiles(min_lat, min_lon, max_lat, max_lon, folder, zoom=19):
    if not os.path.exists(folder):
        os.makedirs(folder)

    mapbox1 = lonlat2mapboxTile([min_lon, min_lat], zoom)
    mapbox2 = lonlat2mapboxTile([max_lon, max_lat], zoom)

    for i in range(mapbox1[0], mapbox2[0] + 1):
        for j in range(mapbox2[1], mapbox1[1] + 1):
            filename = folder + "/%d_%d_%d.jpeg" % (zoom, i, j)
            if not os.path.isfile(filename):
                downloadMapBox(zoom, [i, j], filename)


def GetMapInRect(min_lat, min_lon, max_lat, max_lon, folder="cache/", start_lat=42.1634, start_lon=-71.36,
                 resolution=1024, padding=128, zoom=16, scale=2):
    mapbox1 = lonlat2mapboxTile([min_lon, min_lat], zoom)
    mapbox2 = lonlat2mapboxTile([max_lon, max_lat], zoom)

    ok = True

    print(mapbox1, mapbox2)

    print((mapbox2[0] - mapbox1[0]) * (mapbox1[1] - mapbox2[1]))

    dimx = (mapbox2[0] - mapbox1[0] + 1) * 512  # lon x
    dimy = (mapbox1[1] - mapbox2[1] + 1) * 512  # lat y

    img = np.zeros((dimy, dimx, 3), dtype=np.uint8)

    for i in range(mapbox2[0] - mapbox1[0] + 1):
        if not ok:
            break

        for j in range(mapbox1[1] - mapbox2[1] + 1):
            filename = folder + "/%d_%d_%d.jpeg" % (zoom, i + mapbox1[0], j + mapbox2[1])
            Succ = os.path.isfile(filename)

            if Succ:
                try:
                    subimg = imageio.imread(filename).astype(np.uint8)
                    if subimg.shape[0] != 512 or subimg.shape[1] != 512:
                        subimg = np.array(Image.fromarray(subimg).resize((512, 512), Image.Resampling.LANCZOS))
                except:
                    print("image file is damaged, try to redownload it", filename)
                    Succ = False

            if not Succ:
                Succ = downloadMapBox(zoom, [i + mapbox1[0], j + mapbox2[1]], filename)
                if Succ:
                    subimg = imageio.imread(filename).astype(np.uint8)
                    # 确保subimg是512x512像素
                    if subimg.shape[0] != 512 or subimg.shape[1] != 512:
                        subimg = np.array(Image.fromarray(subimg).resize((512, 512), Image.Resampling.LANCZOS))

            if Succ:
                subimg = imageio.imread(filename).astype(np.uint8)
                if subimg.shape[0] != 512 or subimg.shape[1] != 512:
                    subimg = np.array(Image.fromarray(subimg).resize((512, 512), Image.Resampling.LANCZOS))
                img[j * 512:(j + 1) * 512, i * 512:(i + 1) * 512, :] = subimg

            else:
                ok = False
                break

    x1, y1 = lonlat2TilePos([min_lon, max_lat], zoom)
    x2, y2 = lonlat2TilePos([max_lon, min_lat], zoom)

    # 像素位置
    x2 = x2 + dimx - 512
    y2 = y2 + dimy - 512

    img = img[y1:y2, x1:x2]

    return img, ok



# img, ok = GetMapInRect(45.49066, -122.708558, 45.509092018432014, -122.68226506517134, start_lat = 45.49066, start_lon = -122.708558, zoom=16)

# Image.fromarray(img).save("mapboxtmp.png")


# https://c.tiles.mapbox.com/v4/mapbox.satellite/15/5264/11434@2x.jpg?access_token=pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjaml5MjVyb3MwMWV0M3hxYmUzdGdwbzE4In0.q548FjhsSJzvXsGlPsFxAQ
