# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/17 17:36
@Author : karsten
@File : hangji_locate_without_velocity.py
@Software: PyCharm
============================
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# 对其中一个地震事件而言

import numpy as np
import os
import obspy
import pandas as pd
from pyproj import CRS, Transformer
import locate_fun

###############################
# 读取某个地震事件下所有的数据
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake/2022-07-14T01:44:55.000Z'
stn = obspy.Stream()  # 初始化一个空的Stream对象
for root, dirs, files in os.walk(event_path):
	for file in files:
		file_path = os.path.join(event_path, file)
		stn += obspy.read(file_path)
stp = stn.copy()
stp = locate_fun.waveform_processing(stp)
# 读取台站的位置数据并转换为相对坐标
# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt',
					   header=None, names=['station', 'lon', 'lat'], sep=',')

# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}

stations['station'] = stations['station'].str.split('_').str.get(0)

# 设置投影 - WGS84 经纬度到 UTM
crs_latlon = CRS.from_epsg(4326)  # WGS84
crs_utm = CRS.from_epsg(32648)  # UTM Zone 48, WGS84
transformer = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)

# 转换到UTM并找出参考点
ref_station = '22917'  # 选择一个参考台站
ref_x, ref_y = transformer.transform(station_info[ref_station]['lon'], station_info[ref_station]['lat'])

# 计算相对坐标
rel_coordinates = []
for station, coords in station_info.items():
	x, y = transformer.transform(coords['lon'], coords['lat'])
	rel_coordinates.append([station, x - ref_x, y - ref_y])
rel_coordinates = np.array(rel_coordinates)

###############################
# 基础参数设置
# 假设地震位置为E0
E0 = np.array([50.0, 50.0]).T
V0 = 150
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
# 台站个数
S = len(rel_coordinates)
# 总的方程数量
N = int(S / 2 * (S - 1))
M = 3  # 未知数个数
# 权重矩阵
W = np.zeros((N, N))  # 具体使用互相关大小以及初定误差给定
corr = np.zeros((N, 1))  # 记录互相关大小
sigma_t = 1 / sample_rate  # 用采样间隔来给定
# 记录地震位置的变化
E = [E0]
V = [V0]
# 记录误差的变化
R = [10000]
# 最大循环个数
stop_circle = 10
stop_critical = 0.0001
###############################
# 开始构建矩阵
for circle in range(stop_circle):
	# 重置d和G矩阵
	d = np.zeros((N, 1))
	G = np.zeros((N, M))

	# 构建d和G
	for i in range(len(rel_coordinates)):
		for j in range(i + 1, len(rel_coordinates)):
			# 绝对位置
			cnt = int(
				(i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
			# 对应的台站信息
			station_i = rel_coordinates[i, 0]
			station_j = rel_coordinates[j, 0]
			s_i = np.array([rel_coordinates[i, 1], rel_coordinates[i, 2]], dtype='float64')
			s_j = np.array([rel_coordinates[j, 1], rel_coordinates[j, 2]], dtype='float64')
			# 对应的波形信息
			sw_i = stp.select(station=station_i)
			sw_j = stp.select(station=station_j)
			# 需要判断是否为空才进行下一步
			if len(sw_i) > 0 and len(sw_j) > 0:
				w_i = sw_i[0]
				w_j = sw_j[0]
				maxcor, lagtime = locate_fun.lag_time(w_i, w_j, maxlag, sample_rate)
				# 去除一些互相关系数过小的，认为0.3以下不太行
				if maxcor > 0.3:
					corr[cnt] = maxcor
					W[cnt, cnt] = maxcor / sigma_t
					d[cnt] = lagtime - (locate_fun.dis(E0, s_i) - locate_fun.dis(E0, s_j)) / V0
					G[cnt, 0] = ((E0[0] - s_i[0]) / locate_fun.dis(E0, s_i) - (E0[0] - s_j[0]) / locate_fun.dis(E0, s_j)) / V0
					G[cnt, 1] = ((E0[1] - s_i[1]) / locate_fun.dis(E0, s_i) - (E0[1] - s_j[1]) / locate_fun.dis(E0, s_j)) / V0
					G[cnt, 2] = -1/V0**2 * (locate_fun.dis(E0, s_i) - locate_fun.dis(E0, s_j))

	# 使用 TSVD 或其他方法解方程
	delta_m = np.linalg.pinv(W @ G) @ W @ d  # 使用伪逆来求解
	# 更新误差

	# 更新地震位置估计
	E0 = E0 + delta_m[:2].flatten()  # 确保delta_m是一维的
	V0 = V0 + delta_m[2]
	residual = np.linalg.norm(G @ delta_m - d)
	# 检查误差，如果误差增加，则停止迭代
	if residual > R[circle] or np.abs(residual - R[circle]) < stop_critical:
		break
	else:
		E.append(E0.copy())  # 存储新的位置估计
		V.append(V0.copy())
		R.append(residual.copy())  # 存储新的误差
		# 打印或记录信息
		print(f"Iteration {circle}: Delta M = {delta_m.T}, New E0 = {E0}, New V0 = {V0}, Residual = {residual}")
# 迭代结束
