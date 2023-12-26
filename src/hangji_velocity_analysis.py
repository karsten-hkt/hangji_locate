# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/17 12:17
@Author : karsten
@File : hangji_velocity_analysis.py
@Software: PyCharm
============================
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/16 11:28
@Author : karsten
@File : hangji_locate_grid_search.py
@Software: PyCharm
============================
"""
import numpy as np
import obspy
import pandas as pd
import os
import station_fun
import locate_fun
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#####################################
# 实际观测到地震波并进行预处理
event_name = '01:44:55.000Z'
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake/2022-07-14T01:56:55.000Z'
stn = obspy.Stream()  # 初始化一个空的Stream对象
for root, dirs, files in os.walk(event_path):
	for file in files:
		file_path = os.path.join(event_path, file)
		stn += obspy.read(file_path)
stp = stn.copy()
stp = locate_fun.waveform_processing(stp)

#####################################
# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')

# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}
ref_station = '22917'  # 选择一个参考台站
rel_coordinates = station_fun.rel_coordinates(station_info, ref_station)

#####################################
# 基础参数设置
# 台站个数
S = len(rel_coordinates)
# 总的方程数量
N = int(S / 2 * (S - 1))
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
# 保存各个台站之间的到时差以及台站消息
S_corr = np.zeros((N,9))  # 各个参数为对应的台站1，台站2，台站1位置x,y，台站2位置x,u，到时差，互相关系数，速度

########################################
# 计算网格之间的到时差并保存
for i in range(len(rel_coordinates)):
	for j in range(i + 1, len(rel_coordinates)):
		# 绝对位置
		cnt = int(
			(i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
		# 对应的台站信息
		station_i = rel_coordinates[i, 0]
		station_j = rel_coordinates[j, 0]
		S_corr[cnt,0] = station_i
		S_corr[cnt,1] = station_j
		s_i = np.array([rel_coordinates[i, 1], rel_coordinates[i, 2]], dtype='float64')
		s_j = np.array([rel_coordinates[j, 1], rel_coordinates[j, 2]], dtype='float64')
		S_corr[cnt, 2] = rel_coordinates[i, 1]
		S_corr[cnt, 3] = rel_coordinates[i, 2]
		S_corr[cnt, 4] = rel_coordinates[j, 1]
		S_corr[cnt, 5] = rel_coordinates[j, 2]
		# 对应的波形信息
		sw_i = stp.select(station=station_i)
		sw_j = stp.select(station=station_j)
		# 需要判断是否为空才进行下一步
		if len(sw_i) > 0 and len(sw_j) > 0:
			w_i = sw_i[0]
			w_j = sw_j[0]
			maxcor, lagtime = locate_fun.lag_time(w_i, w_j, maxlag, sample_rate)
			# 去除一些互相关系数过小的，认为0.3以下不太行
			if maxcor > 0.3 and lagtime != 0:
				S_corr[cnt,6] = lagtime
				S_corr[cnt,7] = maxcor
				S_corr[cnt,8] = locate_fun.dis(s_i, s_j) / np.abs(lagtime)
			else:
				S_corr[cnt,6] = np.nan
				S_corr[cnt,7] = np.nan
				S_corr[cnt, 8] = np.nan
		else:
			S_corr[cnt,6] = np.nan
			S_corr[cnt,7] = np.nan
			S_corr[cnt, 8] = np.nan

############################################
# 开始画图
plt.figure(figsize=(12, 8))
# 绘制各个台站
for station in rel_coordinates:
	plt.plot(station[1].astype(float), station[2].astype(float), 'o', markersize=5, color='yellow')
	plt.text(station[1].astype(float), station[2].astype(float), station[0])


# 处理np.nan值
valid_values = S_corr[:, 8][~np.isnan(S_corr[:, 8])]
# 确定颜色映射
norm = plt.Normalize(50, 	1000)
cmap = cm.ScalarMappable(norm=norm, cmap='viridis')
cmap.set_array([])  # 设置颜色映射数组

# 绘制线条，忽略np.nan值
for i in range(len(S_corr)):
	if not np.isnan(S_corr[i, 8]):
		color = cmap.to_rgba(S_corr[i, 8])
		plt.plot([S_corr[i, 2], S_corr[i, 4]], [S_corr[i, 3], S_corr[i, 5]], color=color, linestyle='-')

plt.xlabel('Relative X coordinate (meters)')
plt.ylabel('Relative Y coordinate (meters)')
plt.title('hangji velocity between stations')

# 添加颜色条
# 创建用于颜色条的专用轴
cax = plt.axes([0.92, 0.1, 0.02, 0.8])  # 这里的四个参数分别为 [left, bottom, width, height]，可根据需要调整
plt.colorbar(cmap, cax=cax)


plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/hangji velocity between stations.jpg',dpi=300)