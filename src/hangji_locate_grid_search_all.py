# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/16 15:13
@Author : karsten
@File : hangji_locate_grid_search_all.py
@Software: PyCharm
============================
"""

import numpy as np
import os
import obspy
import pandas as pd
import locate_fun
import station_fun

###############################
# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')
# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}
stations['station'] = stations['station'].str.split('_').str.get(0)
ref_station = '22917'  # 选择一个参考台站
rel_coordinates = station_fun.rel_coordinates(station_info, ref_station)

#####################################
# 基础参数设置
# 假设表层传播的速度为v m/s
v = 200
# 台站个数
S = len(rel_coordinates)
# 总的方程数量
N = int(S / 2 * (S - 1))
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
# 保存每个hangji quake的震动位置
hangji_quake = []
############################
# 构建网格
# 定义搜索区域和网格
grid_size = 1
x_min = 0
y_min = 0
grid_x = np.arange(x_min-60, x_min+60, grid_size)
grid_y = np.arange(y_min-20, y_min+100, grid_size)
grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)

###########################
# 保存各个理论到时差
theritcal_times = locate_fun.theritcal_time(rel_coordinates, grid_points, v)

###############################
# 读取某个地震事件下所有的数据
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake'
# 遍历文件夹，并对每个地震进行逐个定位
for root, dirs, files in os.walk(event_path):
	for dir in dirs:
		full_dir_path = os.path.join(root, dir)
		print('开始处理文件夹：', dir)
		stn = obspy.Stream()
		for file in os.listdir(full_dir_path):
			file_path = os.path.join(full_dir_path, file)
			stn += obspy.read(file_path)
		stp = stn.copy()
		stp = locate_fun.waveform_processing(stp)

		# 保存各个台站之间的到时差以及台站消息
		S_corr = np.zeros((N, 8))  # 各个参数为对应的台站1，台站2，台站1位置x,y，台站2位置x,u，到时差，互相关系数
		# 计算网格之间的到时差并保存
		for i in range(len(rel_coordinates)):
			for j in range(i + 1, len(rel_coordinates)):
				# 绝对位置
				cnt = int(
					(i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(
						rel_coordinates))
				# 对应的台站信息
				station_i = rel_coordinates[i, 0]
				station_j = rel_coordinates[j, 0]
				S_corr[cnt, 0] = station_i
				S_corr[cnt, 1] = station_j
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
					if maxcor > 0.3:
						S_corr[cnt, 6] = lagtime
						S_corr[cnt, 7] = maxcor
					else:
						S_corr[cnt, 6] = np.nan
						S_corr[cnt, 7] = np.nan
				else:
					S_corr[cnt, 6] = np.nan
					S_corr[cnt, 7] = np.nan
		# 与理论到时字典中各个位置进行比较
		obs_arrival = S_corr[:, 6]
		errors = [] # 记录理论的到时

		for point, theritcal_time in theritcal_times.items():
			valid_indices = ~np.isnan(obs_arrival)
			error = np.linalg.norm(theritcal_time[valid_indices] - obs_arrival[valid_indices]) ** 2 / len(
				theritcal_time[valid_indices])
			errors.append(error)
		hangji_quake.append([dir, grid_points[np.argmin(errors)][0], grid_points[np.argmin(errors)][1], errors[np.argmin(errors)]])
		print('最小震中的位置为：', grid_points[np.argmin(errors)])

# 将结果保存在output文件夹下
hangji_quake = pd.DataFrame(hangji_quake, columns=['event_name', 'x', 'y','error'])
hangji_quake.to_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/event_locate_grid_search.csv', index=False)
