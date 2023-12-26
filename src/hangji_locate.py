# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/15 17:08
@Author : karsten
@File : hangji_locate.py
@Software: PyCharm
============================
"""
# 对其中一个地震事件而言

import numpy as np
import os
import obspy
import pandas as pd
from pyproj import CRS, Transformer
from obspy.signal.cross_correlation import correlate

###############################
# 计算两个点之间距离的函数（常用）
def dis(p1, p2):
	'''
	计算两个点之间的距离
	'''
	return np.linalg.norm(p1 - p2)


def waveform_processing(st):
	'''
	对全部stream进行预处理
	滤波5hz以上，去除趋势，去除仪器响应，归一化，去均值
	'''
	# 滤波
	st.filter("highpass", freq=5.0)
	# 归一化
	st.normalize()
	# 去均值
	st.detrend("demean")
	# 去除趋势
	st.detrend("linear")
	# taper
	st.taper(0.05)
	return st


# 计算两个波形之间的互相关函数并返回相对时间
def lag_time(waveform1, waveform2, maxlag, sample_rate):
	'''
	计算两个波形之间的互相关以及延迟时间
	'''
	# 计算互相关函数
	corr = correlate(waveform1, waveform2, maxlag)
	# 找到互相关函数的最大值
	max_corr = np.max(corr)
	max_corr_index = np.argmax(corr)
	# 计算实际的时间延迟
	# 如果max_corr_index大于maxlag，实际延迟应为负值
	lag_samples = max_corr_index - maxlag
	# 假设采样率为 sample_rate，单位为 Hz（每秒样本数）
	lag_time = lag_samples / sample_rate  # 转换为秒
	return max_corr, lag_time


###############################
# 读取某个地震事件下所有的数据
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake/2022-07-14T01:44:55.000Z'
stn = obspy.Stream()  # 初始化一个空的Stream对象
for root, dirs, files in os.walk(event_path):
	for file in files:
		file_path = os.path.join(event_path, file)
		stn += obspy.read(file_path)
stp = stn.copy()
stp = waveform_processing(stp)
# 读取台站的位置数据并转换为相对坐标
# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')

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
# 假设表层传播的速度为v m/s
v = 200
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
# 台站个数
S = len(rel_coordinates)
# 总的方程数量
N = int(S / 2 * (S - 1))
M = 2  # 未知数个数
# 权重矩阵
W = np.zeros((N, N))  # 具体使用互相关大小以及初定误差给定
corr = np.zeros((N, 1))  # 记录互相关大小
sigma_t = 1 / sample_rate  # 用采样间隔来给定
# 记录地震位置的变化
E = []
E.append(E0)
# 记录误差的变化
R = []
R.append(10000)  # 为了和E对齐
# 设定保留的奇异值个数
k = 2
# 最大循环个数
stop_circle = 10
stop_critical = 0.0001
###############################
# 开始构建矩阵
for circle in range(stop_circle):
	# 重置d和G矩阵
	d = np.zeros((N, 1))
	G = np.zeros((N, 2))

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
				maxcor, lagtime = lag_time(w_i, w_j, maxlag, sample_rate)
				# 去除一些互相关系数过小的，认为0.3以下不太行
				if maxcor > 0.3:
					corr[cnt] = maxcor
					W[cnt, cnt] = maxcor / sigma_t
					d[cnt] = lagtime - (dis(E0, s_i) - dis(E0, s_j)) / v
					G[cnt, 0] = ((E0[0] - s_i[0]) / dis(E0, s_i) - (E0[0] - s_j[0]) / dis(E0, s_j)) / v
					G[cnt, 1] = ((E0[1] - s_i[1]) / dis(E0, s_i) - (E0[1] - s_j[1]) / dis(E0, s_j)) / v

	# 使用 TSVD 或其他方法解方程
	delta_m = np.linalg.pinv(W @ G) @ W @ d  # 使用伪逆来求解
	# 更新误差

	# 更新地震位置估计
	E0 = E0 + delta_m.flatten()  # 确保delta_m是一维的
	residual = np.linalg.norm(G @ delta_m - d)
	# 检查误差，如果误差增加，则停止迭代
	if residual > R[circle] or np.abs(residual - R[circle]) < stop_critical:
		break
	else:
		E.append(E0.copy())  # 存储新的位置估计
		R.append(residual.copy())  # 存储新的误差
		# 打印或记录信息
		print(f"Iteration {circle}: Delta M = {delta_m.T}, New E0 = {E0}, Residual = {residual}")
# 迭代结束
