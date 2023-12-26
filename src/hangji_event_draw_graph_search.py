# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/16 16:15
@Author : karsten
@File : hangji_event_draw_graph_search.py.py
@Software: PyCharm
============================
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

# 读取地震数据,注意这个数据已经是在ref_station为参考点的相对坐标
event_locate = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/event_locate_grid_search.csv'
event_locate = pd.read_csv(event_locate, sep=',')
event_locate['event_name'] = event_locate['event_name'].str.split('T').str.get(1)
event_locate_np = np.array(event_locate)

############################
# 构建网格
# 定义搜索区域和网格
grid_size = 1
x_min = 0
y_min = 0
grid_x = np.arange(x_min-60, x_min+60, grid_size)
grid_y = np.arange(y_min-20, y_min+100, grid_size)
grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)

# 开始画图
plt.figure(figsize=(12, 8))

# 绘制各个台站
for station, (x, y) in rel_coordinates.items():
	plt.plot(x, y, 'o', markersize=8, color = 'yellow')
	plt.text(x, y,station)

# 绘制地震
for i in range(len(event_locate)):
	plt.plot(event_locate_np[i,1], event_locate_np[i,2], '*', markersize=8, color = 'red')
	plt.text(event_locate_np[i,1], event_locate_np[i,2], event_locate_np[i,0], fontsize=10)

# 绘制各个网格点
for point in grid_points:
	plt.plot(point[0], point[1], 'o', markersize=2, color = 'gray', alpha = 0.3)

plt.xlabel('Relative X coordinate (meters)')
plt.ylabel('Relative Y coordinate (meters)')
plt.title('Relative event and station positions of stations(22917)')
plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/Relative event and station positions of stations by grid search(22917)_200.jpg',dpi=300)