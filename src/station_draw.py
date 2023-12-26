# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/15 17:06
@Author : karsten
@File : station_draw.py.py
@Software: PyCharm
============================
"""
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')

# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
                for index, row in stations.iterrows()}

# 设置投影 - WGS84 经纬度到 UTM
crs_latlon = CRS.from_epsg(4326)  # WGS84
crs_utm = CRS.from_epsg(32648)    # UTM Zone 48, WGS84
transformer = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)

# 转换到UTM并找出参考点
ref_station = '22917'  # 选择一个参考台站
ref_x, ref_y = transformer.transform(station_info[ref_station]['lon'], station_info[ref_station]['lat'])

# 计算相对坐标
rel_coordinates = {}
for station, coords in station_info.items():
    x, y = transformer.transform(coords['lon'], coords['lat'])
    rel_coordinates[station] = (x - ref_x, y - ref_y)

# 假设一地震位置发生在E0
E0 = [20,20]
# 将地震到达不同台站对之间的路径画出来
# 绘图
plt.figure(figsize=(12, 8))

# 绘制各个台站的连线
# 使用循环遍历字典中的每个坐标对，并将它们连起来
for station1, (x1, y1) in rel_coordinates.items():
    for station2, (x2, y2) in rel_coordinates.items():
        if station1 != station2:
            plt.plot([x1, x2], [y1, y2], color = 'gray', linestyle='-')
# 绘制各个台站
for station, (x, y) in rel_coordinates.items():
    plt.plot(x, y, 'o', markersize=5, color = 'r')
    # 将地震和各个台站连起来
    plt.plot([E0[0], x], [E0[1], y], color = 'y', linestyle='-')
    plt.text(x, y,station)

# 绘制地震
plt.plot(E0[0], E0[1], '*', markersize=12, color='y', label = 'hangji_quake')

plt.xlabel('Relative X coordinate (meters)')
plt.ylabel('Relative Y coordinate (meters)')
plt.title('Relative Positions of Stations')
plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/Relative Positions of Stations.jpg')
