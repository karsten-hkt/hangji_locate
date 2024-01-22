# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/21 14:30
@Author : karsten
@File : hangji_locate_all_variance_draw.py.py
@Software: PyCharm
============================
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
from matplotlib.patches import Ellipse
font = {
'weight' : 'normal',
'size'   : 15,
        }
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
event_locate = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/event_locate_variance_0.01.csv'
event_locate = pd.read_csv(event_locate, sep=',')
event_locate['event_name'] = event_locate['event_name'].str.split('T').str.get(1)
event_locate_np = np.array(event_locate)
# 开始画图
plt.figure(figsize=(12, 8))

# 绘制各个台站
for station, (x, y) in rel_coordinates.items():
	plt.plot(x, y, 'o', markersize=5, color = 'r')
	plt.text(x, y,station)

# 绘制地震
for i in range(len(event_locate)):
	if event_locate_np[i, 3] > 80 or event_locate_np[i, 4] > 80:
		plt.plot(event_locate_np[i, 1], event_locate_np[i, 2], 'x', markersize=10, color='blue')
		plt.text(event_locate_np[i, 1]-5, event_locate_np[i, 2]-5, event_locate_np[i, 0]+'_wrong', fontsize=8)
	else:
		plt.plot(event_locate_np[i, 1], event_locate_np[i, 2], 'o', markersize=10, color='lightblue', alpha=0.5)
		plt.text(event_locate_np[i, 1]-5, event_locate_np[i, 2]-5, event_locate_np[i, 0], fontsize=8)
		# 计算椭圆的宽度和高度（方差的平方根的两倍作为椭圆的轴长）
		ellipse_x = 2 * np.sqrt(event_locate_np[i, 3])
		ellipse_y = 2 * np.sqrt(event_locate_np[i, 4])
		# 添加椭圆
		ellipse = Ellipse((event_locate_np[i, 1], event_locate_np[i, 2]), width=ellipse_x, height=ellipse_y,
						  edgecolor='b',
						  facecolor='none')
		plt.gca().add_patch(ellipse)
# 设置x，y坐标值范围

plt.xlim(-60,60)
plt.ylim(-20,100)
plt.xlabel('Relative X coordinate (meters)',font)
plt.ylabel('Relative Y coordinate (meters)',font)
plt.title('Relative event and station positions of stations(22917)',font)
plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/Relative event and station positions of stations(22917)_200_sigma_0.005.jpg',dpi=300)