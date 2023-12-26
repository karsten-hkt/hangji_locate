# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/23 15:04
@Author : karsten
@File : hangji_locate_Tikhonov_Regularization_variance_draw.py
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
m = np.load('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/m_locate.npy')
m_variance = np.load('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/m_variance.npy')
event_name = np.load('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/event_name.npy')
event_name = [timestamp.split('T')[1] for timestamp in event_name]
###############################
# 开始画图
plt.figure(figsize=(12, 8))

# 绘制各个台站
for station, (x, y) in rel_coordinates.items():
	plt.plot(x, y, 'o', markersize=5, color = 'r')

K = 20 #总地震个数
# 绘制地震
for i in range(K):
	if m_variance[3*i, 3*i] > 80 or m_variance[3*i+1, 3*i+1] > 80:
		plt.plot(m[3*i], m[3*i+1], 'x', markersize=5, color='blue')
		plt.text(m[3*i], m[3*i+1], event_name[i]+'_wrong'+f' v={m[3*i+2]}', fontsize=8)
	else:
		plt.plot(m[3*i], m[3*i+1], 'o', markersize=5, color='lightblue', alpha=0.5)
		plt.text(m[3*i], m[3*i+1],  event_name[i]+f' v={m[3*i+2]}', fontsize=8)
		# 计算椭圆的宽度和高度（方差的平方根的两倍作为椭圆的轴长）
		ellipse_x = 2 * np.sqrt(m_variance[3*i, 3*i])
		ellipse_y = 2 * np.sqrt(m_variance[3*i+1, 3*i+1])
		# 添加椭圆
		ellipse = Ellipse((m[3*i], m[3*i+1]), width=ellipse_x, height=ellipse_y,
						  edgecolor='b',
						  facecolor='none')
		plt.gca().add_patch(ellipse)
plt.xlabel('Relative X coordinate (meters)')
plt.ylabel('Relative Y coordinate (meters)')
plt.title('Relative event and station positions of stations(22917)')
plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/Tikohonov_Regulazation_with_error_0.01.jpg',dpi=300)